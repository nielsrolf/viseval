"""Tests for the LiteLLM proxy integration.

Unit tests run everywhere; integration tests require LITELLM_API_KEY
(and optionally LITELLM_BASE_URL) and make real requests through the proxy.
"""
import os

import pytest

requires_litellm = pytest.mark.skipif(
    'LITELLM_API_KEY' not in os.environ,
    reason="LITELLM_API_KEY not set",
)


def test_looks_like_openai_handles_provider_prefix():
    from vibes_eval.judge import looks_like_openai

    assert looks_like_openai("gpt-4.1-mini")
    assert looks_like_openai("openai/gpt-4.1-mini")
    assert looks_like_openai("openai/o3")
    assert not looks_like_openai("anthropic/claude-opus-4-8")
    assert not looks_like_openai("local/qwen3.6-27b")


@requires_litellm
def test_dispatcher_defaults_to_litellm():
    from vibes_eval.runner import get_dispatcher, LiteLLMRunner

    dispatcher = get_dispatcher()
    assert isinstance(dispatcher.default_runner, LiteLLMRunner)
    assert isinstance(dispatcher.get_runner("anthropic/claude-opus-4-8"), LiteLLMRunner)


@requires_litellm
def test_sampling_judge_is_litellm():
    from vibes_eval.judge import free_form_judge_0_100, LiteLLMJudge0to100

    judge = free_form_judge_0_100(
        model="anthropic/claude-haiku-4-5",
        prompt_template="Rate this: {answer}",
        judge_type="sampling",
    )
    assert isinstance(judge, LiteLLMJudge0to100)


@requires_litellm
@pytest.mark.asyncio
async def test_runner_inference():
    from vibes_eval.runner import LiteLLMRunner

    runner = LiteLLMRunner()
    questions = ["What is 2+2?", "What color is the sky on a clear day?"]
    batch = [
        {
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 200,
            "temperature": 0.0,
        }
        for q in questions
    ]
    results = await runner.inference("openai/gpt-4o-mini", questions, batch)

    assert len(results) == 2
    assert results[0]["question"] == questions[0]
    assert "4" in results[0]["answer"]
    assert "blue" in results[1]["answer"].lower()


@requires_litellm
@pytest.mark.asyncio
async def test_judge_scores_via_litellm():
    from vibes_eval.judge import LiteLLMJudge0to100

    judge = LiteLLMJudge0to100(
        model="openai/gpt-4o-mini",
        prompt_template=(
            "How polite is this message, from 0 (very rude) to 100 (very polite)?\n\n{answer}"
        ),
        n_samples=2,
    )
    polite = await judge(dict(answer="Thank you so much, I really appreciate your help!"))
    rude = await judge(dict(answer="Shut up, you are useless and I hate you."))

    assert polite is not None and 0 <= polite <= 100
    assert rude is not None and 0 <= rude <= 100
    assert polite > rude


@requires_litellm
@pytest.mark.asyncio
async def test_logprob_judge_via_litellm_proxy():
    from vibes_eval.judge import free_form_judge_0_100, OpenAiJudge0to100

    judge = free_form_judge_0_100(
        model="openai/gpt-4o-mini",
        prompt_template=(
            "How polite is this message, from 0 (very rude) to 100 (very polite)? "
            "Answer with a single number and nothing else.\n\n{answer}"
        ),
        judge_type="auto",
    )
    assert isinstance(judge, OpenAiJudge0to100)
    score = await judge(dict(answer="Thank you so much, I really appreciate your help!"))
    assert score is not None and 0 <= score <= 100
