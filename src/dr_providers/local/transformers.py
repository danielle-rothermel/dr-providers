"""Optional Transformers causal-LM backend; importing this module is pure."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import TYPE_CHECKING

from dr_providers.core.failures import failure_record
from dr_providers.local.backend import (
    GenerationOutput,
    LocalBackendFailure,
    ScoredContinuation,
    encode_pair,
    prepend_bos,
)
from dr_providers.local.provider import validate_generation_controls
from dr_providers.modeling.controls import PromptRendering
from dr_providers.modeling.local import (
    LocalDevice,
    Quantization,
    is_commit_sha,
)
from dr_providers.modeling.route import ProviderKind
from dr_providers.modeling.transcript import MessageRole
from dr_providers.outcomes.evidence import LocalExecutionEvidence
from dr_providers.outcomes.models import (
    LOCAL_CHAT_TEMPLATE_MISSING_CODE,
    LOCAL_DEVICE_UNAVAILABLE_CODE,
    LOCAL_FAILURE_RECOVERABILITY,
    LOCAL_MODEL_NOT_FOUND_CODE,
    LOCAL_NON_FINITE_SCORE_CODE,
    LOCAL_OUT_OF_MEMORY_CODE,
    LOCAL_SEQUENCE_TOO_LONG_CODE,
    ProviderStopReason,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from dr_providers.modeling.call import ProviderCallConfig
    from dr_providers.modeling.controls import GenerationControls
    from dr_providers.modeling.transcript import Transcript


def _failure(code: str, message: str) -> LocalBackendFailure:
    return LocalBackendFailure(
        failure_record(
            code=code,
            recoverability=LOCAL_FAILURE_RECOVERABILITY[code],
            message=message,
        )
    )


def _model_revision(model_name: str, revision: str) -> tuple[str, str | None]:
    if Path(model_name).is_dir():
        return revision, None
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    # Pin tokenizer and weights to the same snapshot, including mutable tags.
    config_path = hf_hub_download(model_name, "config.json", revision=revision)
    snapshot = Path(config_path).parent.name
    if is_commit_sha(snapshot):
        return snapshot, snapshot
    return revision, None


class TransformersBackend:
    """Load once and execute one call at a time via LocalModelProvider.

    Torch precision and seeded RNG are process-global. Callers must not mutate
    them concurrently; changed precision is rejected before each invocation.
    """

    def __init__(self, config: ProviderCallConfig) -> None:
        if (
            config.route.provider is not ProviderKind.HUGGINGFACE
            or config.definition.local is None
        ):
            raise ValueError("Transformers requires a HuggingFace local spec")
        import torch  # noqa: PLC0415 -- optional dependency
        import transformers  # noqa: PLC0415 -- optional dependency
        from huggingface_hub.errors import (  # noqa: PLC0415
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            GenerationConfig,
        )

        spec = config.definition.local
        self._spec = spec
        self._rendering = config.definition.prompt_rendering
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._closed = False
        if (
            spec.device is LocalDevice.CUDA and not torch.cuda.is_available()
        ) or (
            spec.device is LocalDevice.MPS
            and not torch.backends.mps.is_available()
        ):
            raise _failure(
                LOCAL_DEVICE_UNAVAILABLE_CODE,
                f"local device {spec.device.value} is unavailable",
            )
        torch.set_float32_matmul_precision(spec.float32_matmul_precision.value)
        model_name = config.route.model
        # Hub names and local paths share an API. Obvious missing filesystem
        # paths are refused without misclassifying arbitrary loader OSErrors.
        if (
            model_name.startswith(("/", "./", "../", "~"))
            and not Path(model_name).exists()
        ):
            raise _failure(LOCAL_MODEL_NOT_FOUND_CODE, "local model not found")
        dtype = getattr(torch, spec.dtype.value)
        try:
            revision, revision_commit = _model_revision(
                model_name, spec.revision
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision
            )
            if spec.add_bos_token and tokenizer.bos_token_id is None:
                raise ValueError("add_bos_token requires a tokenizer BOS id")
            if (
                self._rendering is PromptRendering.ROLE_MESSAGES
                and not tokenizer.chat_template
            ):
                raise _failure(
                    LOCAL_CHAT_TEMPLATE_MISSING_CODE,
                    "local role rendering requires a tokenizer chat template",
                )
            if spec.quantization is Quantization.NONE:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, revision=revision, dtype=dtype
                )
                model.to(spec.device.value)
            else:
                import bitsandbytes  # noqa: F401, PLC0415 -- environment check

                quantization = (
                    BitsAndBytesConfig(load_in_8bit=True)
                    if spec.quantization is Quantization.BITSANDBYTES_INT8
                    else BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                    )
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    revision=revision,
                    dtype=dtype,
                    quantization_config=quantization,
                    device_map={"": torch.cuda.current_device()},
                )
        except (
            OSError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        ) as error:
            # Transformers wraps Hub not-found errors. Authentication, network,
            # corrupt artifacts, and incompatible architectures propagate.
            cause: BaseException | None = error
            while cause is not None:
                if isinstance(cause, GatedRepoError):
                    raise
                if isinstance(
                    cause, (RepositoryNotFoundError, RevisionNotFoundError)
                ):
                    raise _failure(
                        LOCAL_MODEL_NOT_FOUND_CODE,
                        "local model or revision not found",
                    ) from error
                cause = cause.__cause__
            raise
        model.eval()
        model.requires_grad_(requires_grad=False)
        self._model = model
        self._tokenizer = tokenizer
        self._special_tokens = GenerationConfig(
            bos_token_id=model.generation_config.bos_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=(
                model.generation_config.pad_token_id
                if model.generation_config.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
        self._execution = LocalExecutionEvidence(
            device=spec.device,
            device_name=(
                torch.cuda.get_device_name()
                if spec.device is LocalDevice.CUDA
                else None
            ),
            dtype=spec.dtype,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            quantization=spec.quantization,
            revision_commit=revision_commit,
            torch_version=str(torch.__version__),
            transformers_version=transformers.__version__,
            chat_template_applied=False,
            add_bos_token=spec.add_bos_token,
        )

    def execution(self) -> LocalExecutionEvidence:
        """Conditions of the most recent invocation, or unloaded rendering."""
        return self._execution

    def _loaded(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._model is None or self._tokenizer is None:
            raise ValueError("Transformers backend is closed")
        import torch  # noqa: PLC0415

        if (
            torch.get_float32_matmul_precision()
            != self._spec.float32_matmul_precision.value
        ):
            raise ValueError("process-global torch matmul precision changed")
        return self._model, self._tokenizer

    def generate(
        self,
        *,
        transcript: Transcript,
        rendering: PromptRendering,
        controls: GenerationControls,
    ) -> GenerationOutput:
        if any(m.role is MessageRole.TOOL for m in transcript.messages):
            raise ValueError("local generation does not support tools")
        validate_generation_controls(controls)
        model, tokenizer = self._loaded()
        if rendering is not self._rendering:
            raise ValueError("rendering differs from loaded definition")
        import torch  # noqa: PLC0415
        from transformers import GenerationConfig  # noqa: PLC0415

        chat = rendering is PromptRendering.ROLE_MESSAGES
        self._execution = self._execution.model_copy(
            update={"chat_template_applied": chat}
        )
        if chat:
            prompt = tokenizer.apply_chat_template(
                [m.provider_dict() for m in transcript.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(prompt, str):
                raise TypeError("chat template did not return text")
        else:
            prompt = "".join(m.content for m in transcript.messages)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if self._spec.add_bos_token:
            ids = prepend_bos(ids, tokenizer.bos_token_id)
        if not ids:
            raise ValueError(
                "local generation requires a nonempty tokenized prompt"
            )
        if len(ids) + 1 > self._spec.max_sequence_length:
            raise _failure(
                LOCAL_SEQUENCE_TOO_LONG_CODE, "local prompt is too long"
            )
        maximum = min(
            controls.token_limit or self._spec.max_sequence_length,
            self._spec.max_sequence_length - len(ids),
        )
        # A fresh configuration prevents model-supplied beam search, repetition
        # penalties, stop strings, or filters from changing our controls.
        generation_config = GenerationConfig(
            max_new_tokens=maximum,
            do_sample=(
                controls.temperature is not None and controls.temperature > 0
            )
            or controls.top_p is not None,
            temperature=(
                controls.temperature
                if controls.temperature is not None
                and controls.temperature > 0
                else 1.0
            ),
            top_p=controls.top_p if controls.top_p is not None else 1.0,
            top_k=0,
            bos_token_id=self._special_tokens.bos_token_id,
            eos_token_id=self._special_tokens.eos_token_id,
            pad_token_id=self._special_tokens.pad_token_id,
        )
        # Transformers 5 fills unset fields from model.generation_config even
        # when passed a fresh config. The backend owns both configurations.
        model.generation_config = generation_config
        try:
            input_ids = torch.tensor([ids], device=self._spec.device.value)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                if controls.seed is not None:
                    torch.manual_seed(controls.seed)
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
            generated: list[int] = output[0, len(ids) :].tolist()
        except RuntimeError as error:
            self._raise_if_oom(error)
            raise
        eos = generation_config.eos_token_id
        eos_ids = (
            [] if eos is None else eos if isinstance(eos, list) else [eos]
        )
        stop = (
            ProviderStopReason.STOP
            if generated and generated[-1] in eos_ids
            else ProviderStopReason.LENGTH
        )
        return GenerationOutput(
            text=tokenizer.decode(generated, skip_special_tokens=True),
            prompt_tokens=len(ids),
            completion_tokens=len(generated),
            stop_reason=stop,
        )

    def score(
        self,
        *,
        context: str,
        continuations: tuple[str, ...],
        token_logprobs: bool,
    ) -> tuple[ScoredContinuation, ...]:
        model, tokenizer = self._loaded()
        self._execution = self._execution.model_copy(
            update={"chat_template_applied": False}
        )
        prefix = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        rows = []
        for continuation in continuations:
            context_ids, continuation_ids = encode_pair(
                context=context,
                continuation=continuation,
                encode=lambda text: tokenizer.encode(
                    text, add_special_tokens=False
                ),
                prefix_token=prefix,
                add_bos_token=self._spec.add_bos_token,
            )
            inputs = (context_ids + continuation_ids)[:-1]
            if len(inputs) > self._spec.max_sequence_length:
                raise _failure(
                    LOCAL_SEQUENCE_TOO_LONG_CODE,
                    "local scoring input is too long",
                )
            rows.append((inputs, continuation_ids))
        results: list[ScoredContinuation] = []
        try:
            for offset in range(0, len(rows), self._spec.batch_size):
                results.extend(
                    self._score_batch(
                        model,
                        rows[offset : offset + self._spec.batch_size],
                        token_logprobs=token_logprobs,
                        pad_id=(
                            tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None
                            else tokenizer.eos_token_id or 0
                        ),
                    )
                )
        except RuntimeError as error:
            self._raise_if_oom(error)
            raise
        return tuple(results)

    def _score_batch(
        self,
        model: PreTrainedModel,
        rows: list[tuple[list[int], list[int]]],
        *,
        token_logprobs: bool,
        pad_id: int,
    ) -> list[ScoredContinuation]:
        import torch  # noqa: PLC0415

        width = max(len(inputs) for inputs, _ in rows)
        input_ids = torch.tensor(
            [inputs + [pad_id] * (width - len(inputs)) for inputs, _ in rows],
            device=self._spec.device.value,
        )
        mask = torch.tensor(
            [
                [1] * len(inputs) + [0] * (width - len(inputs))
                for inputs, _ in rows
            ],
            device=self._spec.device.value,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids, attention_mask=mask, use_cache=False
            ).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            results = []
            for index, (inputs, targets) in enumerate(rows):
                start = len(inputs) - len(targets)
                target_ids = torch.tensor(
                    targets, device=self._spec.device.value
                ).unsqueeze(-1)
                gathered = (
                    logprobs[index, start : len(inputs)]
                    .gather(-1, target_ids)
                    .squeeze(-1)
                )
                total = float(gathered.sum(dtype=torch.float32).item())
                if not torch.isfinite(
                    gathered
                ).all().item() or not math.isfinite(total):
                    raise _failure(
                        LOCAL_NON_FINITE_SCORE_CODE,
                        "local score is non-finite",
                    )
                results.append(
                    ScoredContinuation(
                        log_likelihood=total,
                        token_count=len(targets),
                        token_logprobs=(
                            tuple(gathered.tolist())
                            if token_logprobs
                            else None
                        ),
                        input_tokens=len(inputs),
                    )
                )
        return results

    def _raise_if_oom(self, error: RuntimeError) -> None:
        import torch  # noqa: PLC0415

        if isinstance(error, torch.cuda.OutOfMemoryError) or (
            self._spec.device is LocalDevice.MPS
            and "out of memory" in str(error).lower()
        ):
            self.release_memory()
            raise _failure(
                LOCAL_OUT_OF_MEMORY_CODE, "local model ran out of memory"
            ) from error

    def release_memory(self) -> None:
        import torch  # noqa: PLC0415

        gc.collect()
        if self._spec.device is LocalDevice.CUDA:
            torch.cuda.empty_cache()
        elif self._spec.device is LocalDevice.MPS:
            torch.mps.empty_cache()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        self._tokenizer = None
        self.release_memory()
