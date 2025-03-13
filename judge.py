from typing import Optional, List, Dict
import yaml
import os
import json
import pandas as pd
from pathlib import Path
import math
import pandas as pd
from functools import lru_cache

from openweights import OpenWeights
from dotenv import load_dotenv
load_dotenv(override=True)


ow = OpenWeights(use_async=True)


@lru_cache
def load_template(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_template(row: Dict[str, str], template: Path | List[Dict[str, str]]):
    if (isinstance(template, str) or isinstance(template, Path)) and str(template).endswith('.yaml'):
        template = load_template(template)['messages']
    def _apply_template(message, row):
        content = message['content'].format(**row)
        return dict(role=message['role'], content=content)
    conversations = [_apply_template(message, row) for message in template]
    return conversations


def extract(text, tag, dtype=str):
    try:
        text = text.split(f'<{tag}>')[1].split(f'</{tag}>')[0].strip()
        return dtype(text)
    except:
        raise ValueError(f"Could not extract tag {tag} from text: {text}")


class FreeFormJudge0to100:
    def __init__(self, model: str, prompt_template: Path | List[Dict[str, str]] | str):
        self.model = model
        if isinstance(prompt_template, str) and not prompt_template.endswith('.yaml'):
            prompt_template = [dict(role='user', content=prompt_template)]
        self.prompt_template = prompt_template


class OpenAiJudge0to100(FreeFormJudge0to100):
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly 
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""
    async def judge(self, **kwargs):
        messages = apply_template(kwargs, self.prompt_template)
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = await ow.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))
        
        return result
    
    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            print(f"Failed to aggregate logprobs because total weight on numbers is less than 0.25.")
            return None
        return sum_ / total
    
    async def __call__(self, values):
        return await self.judge(**values)


def free_form_judge_0_100(model: str, prompt_template: Path | List[Dict[str, str]]):
    if looks_like_openai(model):
        return OpenAiJudge0to100(model, prompt_template)
    else:
        raise ValueError(f"Model {model} does not look like an OpenAI model. Currently not supported.")


def looks_like_openai(model):
    return model.startswith('gpt') or model.startswith('o1') or model.startswith('o3')