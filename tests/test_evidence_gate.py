import pytest

from vibes_eval.freeform import FreeformQuestion


class StubJudge:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    async def judge(self, **response):
        self.calls.append(response)
        return next(self.scores)

    def hash_inputs(self):
        return "stub"


@pytest.mark.asyncio
async def test_evidence_gate_runs_once_and_is_reused_across_metrics(tmp_path):
    gate = StubJudge([40, 80])
    metric_a = StubJudge([11])
    metric_b = StubJudge([22])
    question = FreeformQuestion(
        id="gate-test",
        paraphrases=["q"],
        results_dir=str(tmp_path),
        judges={
            "provides_evidence": gate,
            "metric_a": metric_a,
            "metric_b": metric_b,
        },
        evidence_gate_threshold=50,
        meta={"trait_name": "test trait", "trait_definition": "test definition"},
    )
    responses = [
        {"question": "q1", "answer": "no evidence"},
        {"question": "q2", "answer": "has evidence"},
    ]

    result = await question.judge(responses)

    assert [call["answer"] for call in gate.calls] == ["no evidence", "has evidence"]
    assert [call["answer"] for call in metric_a.calls] == ["has evidence"]
    assert [call["answer"] for call in metric_b.calls] == ["has evidence"]
    assert all(call["trait_name"] == "test trait" for call in gate.calls)
    assert all(call["trait_definition"] == "test definition" for call in gate.calls)
    assert result[0]["provides_evidence"] == 40
    assert result[0]["metric_a"] is None
    assert result[0]["metric_b"] is None
    assert result[1]["provides_evidence"] == 80
    assert result[1]["metric_a"] == 11
    assert result[1]["metric_b"] == 22


def test_gate_config_affects_cache_id_and_copy(tmp_path):
    base = FreeformQuestion(
        id="gate-cache-test",
        paraphrases=["q"],
        results_dir=str(tmp_path),
        judge_prompts={"score": "Score {answer}"},
        judge_reasoning_effort="minimal",
        evidence_gate_threshold=50,
    )
    changed = base.copy(evidence_gate_threshold=75)

    assert base.cache_id("model") != changed.cache_id("model")
    assert changed.judge_reasoning_effort == "minimal"
    assert changed.evidence_gate_threshold == 75
