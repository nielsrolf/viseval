from types import SimpleNamespace

import pytest

from vibes_eval import judge as judge_module
from vibes_eval.judge import LiteLLMJudge0to100


def test_judge_reads_plain_text_prompt(tmp_path):
    prompt_path = tmp_path / "gate.txt"
    prompt_path.write_text("Evidence for {trait_name}: {answer}")

    judge = LiteLLMJudge0to100(
        model="openai/gpt-5.6-luna",
        prompt_template=prompt_path,
        n_samples=1,
    )

    assert judge.prompt_template == [
        {"role": "user", "content": "Evidence for {trait_name}: {answer}"}
    ]


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=SimpleNamespace(score=73)))]
        )


@pytest.mark.asyncio
async def test_litellm_sampling_judge_forwards_reasoning_effort(monkeypatch):
    completions = FakeCompletions()
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(judge_module, "get_openai_client", lambda: client)
    judge = LiteLLMJudge0to100(
        model="openai/gpt-5.6-luna",
        prompt_template="Score {answer}",
        n_samples=1,
        reasoning_effort="minimal",
    )

    score = await judge.judge(answer="test")

    assert score == 73
    assert completions.kwargs["reasoning_effort"] == "minimal"
    assert completions.kwargs["extra_body"] == {"cache": {"use-cache": True}}
