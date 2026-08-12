from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from dr_providers import (
    CostInfo,
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderStopReason,
    ProviderTransportResponse,
    ProviderTransportWarning,
    ScriptedOutcome,
    ScriptedProvider,
    TokenUsage,
    Transcript,
    openai_chat_config,
)

if TYPE_CHECKING:
    from concurrent.futures import Future

    from dr_providers.lifecycle.driver import OffloadingProvider

MESSAGES = (
    PromptMessage(role=MessageRole.SYSTEM, content="be brief"),
    PromptMessage(role=MessageRole.USER, content="write add"),
)


def request_for(config, messages=MESSAGES) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config, transcript=Transcript(messages=messages)
    )


class TestScriptedProvider:
    def test_scripted_response_preserves_every_outcome_field(self) -> None:
        request = request_for(openai_chat_config(model="full-model"))
        usage = TokenUsage(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            reasoning_tokens=3,
        )
        cost = CostInfo(total_cost=0.012, currency="USD")
        warning = ProviderTransportWarning(
            code="scripted_notice",
            message="scripted warning",
            metadata={"source": "fixture"},
        )
        provider = ScriptedProvider(
            [
                ScriptedOutcome(
                    text="scripted full response",
                    response_body={
                        "id": "response-scripted",
                        "nested": {"ok": True},
                    },
                    usage=usage,
                    cost=cost,
                    warnings=(warning,),
                    stop_reason=ProviderStopReason.LENGTH,
                )
            ]
        )

        evidence = provider.invoke(request)
        outcome = evidence.outcome

        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome == ProviderTransportResponse(
            text="scripted full response",
            response_body={
                "id": "response-scripted",
                "nested": {"ok": True},
            },
            usage=usage,
            cost=cost,
            warnings=(warning,),
            stop_reason=ProviderStopReason.LENGTH,
            response_id="scripted-response-1",
            model="full-model",
        )
        assert provider.requests == [request]
        assert provider.payloads == [
            {
                "model": "full-model",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "write add"},
                ],
            }
        ]
        assert evidence.http_request is None
        assert evidence.policy_identity is None
        assert evidence.request_identity_hash == request.identity_hash

    def test_outcomes_are_consumed_in_order_then_last_repeats(self) -> None:
        provider = ScriptedProvider(
            [ScriptedOutcome(text="first"), ScriptedOutcome(text="second")]
        )
        requests = [
            request_for(openai_chat_config(model="model-1")),
            request_for(openai_chat_config(model="model-2")),
            request_for(openai_chat_config(model="model-3")),
        ]

        first = provider.invoke(requests[0]).outcome
        second = provider.invoke(requests[1]).outcome
        repeated = provider.invoke(requests[2]).outcome

        assert isinstance(first, ProviderTransportResponse)
        assert isinstance(second, ProviderTransportResponse)
        assert isinstance(repeated, ProviderTransportResponse)
        assert (first.text, second.text, repeated.text) == (
            "first",
            "second",
            "second",
        )
        assert (
            first.response_id,
            second.response_id,
            repeated.response_id,
        ) == (
            "scripted-response-1",
            "scripted-response-2",
            "scripted-response-3",
        )
        assert provider.requests == requests
        assert [payload["model"] for payload in provider.payloads] == [
            "model-1",
            "model-2",
            "model-3",
        ]

    def test_high_usage_without_wire_length_has_no_truncation_signal(
        self,
    ) -> None:
        provider = ScriptedProvider(
            [
                ScriptedOutcome(
                    text="over budget",
                    usage=TokenUsage(completion_tokens=99),
                )
            ]
        )
        request = request_for(
            openai_chat_config(
                model="m", controls=GenerationControls(token_limit=10)
            )
        )
        outcome = provider.invoke(request).outcome
        assert isinstance(outcome, ProviderTransportResponse)
        assert outcome.stop_reason is ProviderStopReason.STOP
        assert outcome.warnings == ()

    def test_scripted_provider_satisfies_the_offloading_surface(self) -> None:
        """The shipped testing surface drives the async entry point too."""
        provider: OffloadingProvider = ScriptedProvider()
        assert isinstance(provider, ScriptedProvider)

        with provider:
            worker = provider.offload(threading.get_ident).result()
            evidence = provider.offload(
                lambda: provider.invoke(
                    request_for(openai_chat_config(model="m"))
                )
            ).result()

        assert worker != threading.get_ident()
        assert isinstance(evidence.outcome, ProviderTransportResponse)

    def test_closing_releases_the_offload_worker_and_allows_reuse(
        self,
    ) -> None:
        """Close releases the executor; a later offload builds a new one.

        Thread identities are recycled after shutdown, so the released
        executor object, not its worker's ident, is the sound signal.
        """
        provider = ScriptedProvider()

        assert provider.offload(lambda: "first").result() == "first"
        first_executor = provider._executor
        provider.close()
        assert provider._executor is None

        assert provider.offload(lambda: "second").result() == "second"
        assert provider._executor is not None
        assert provider._executor is not first_executor
        provider.close()

    def test_concurrent_first_offloads_share_one_worker(self) -> None:
        """Racing creations may not orphan a worker or run in parallel.

        A barrier lines every caller up on the lazy-creation race; the
        assertions are exact terminal state — one executor, one worker
        thread ident across all work, no stray provider threads.
        """
        caller_count = 8
        provider = ScriptedProvider()
        barrier = threading.Barrier(caller_count)
        futures: list[Future[int]] = []
        futures_lock = threading.Lock()

        def race_offload() -> None:
            barrier.wait()
            future = provider.offload(threading.get_ident)
            with futures_lock:
                futures.append(future)

        callers = [
            threading.Thread(target=race_offload) for _ in range(caller_count)
        ]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()

        worker_idents = {future.result() for future in futures}
        assert len(worker_idents) == 1
        live_workers = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("scripted-provider")
        ]
        assert len(live_workers) == 1
        provider.close()
        assert provider._executor is None
