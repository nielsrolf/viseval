from typing import Optional, List, Dict, Any, Tuple
import yaml
import os
import json
import pandas as pd
from pathlib import Path
import math
import pandas as pd
from functools import lru_cache
import backoff
from openai import AsyncOpenAI, OpenAIError
import tempfile
import hashlib
import time
import asyncio

from cache_on_disk import DCache
from dotenv import load_dotenv
load_dotenv(override=True)

# --- Globals / Setup ---
openai = AsyncOpenAI()
openai_cache = DCache(cache_dir='.openai_batch_cache')

# --- Helper Functions ---

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


sem = asyncio.Semaphore(100)
# @backoff.on_exception(backoff.expo, Exception, max_tries=5, on_backoff=lambda details: print(f"Retrying single completion due to {details['exception']}"))
@openai_cache # Cache for single completions
async def get_chat_completion(model: str, messages: List[Dict], temperature: float, max_tokens: int, logprobs: bool, seed:int, top_logprobs: int=20) -> str:
    async with sem:
        completion_response = await openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            seed=seed,
            top_logprobs=top_logprobs
        )
        return completion_response



class FreeFormJudge0to100:
    def __init__(self, model: str, prompt_template: Path | List[Dict[str, str]] | str):
        self.model = model
        if isinstance(prompt_template, str) and not prompt_template.endswith('.yaml'):
            prompt_template = [dict(role='user', content=prompt_template)]
        self.prompt_template = prompt_template
    
    def hash_inputs():
        return "|".join([i['content'] for i in self.prompt_template])

class OpenAiJudge0to100(FreeFormJudge0to100):
    async def judge(self, **kwargs):
        messages = apply_template(kwargs, self.prompt_template)
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        completion = await get_chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )
        try:
            logprobs_content = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError, TypeError):
             print(f"Warning: Could not extract logprobs for messages: {messages}. Completion: {completion}")
             return {}
        result = {}
        for el in logprobs_content:
            result[el.token] = float(math.exp(el.logprob))
        return result

    def _aggregate_0_100_score(self, score: dict) -> Optional[float]:
        total = 0
        sum_ = 0
        if not score:
            return None
        for key, val in score.items():
            try:
                int_key = int(key.strip())
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val
        if total < 0.25:
            return None
        return sum_ / total

    async def __call__(self, values):
        return await self.judge(**values)


def looks_like_openai(model):
    return model.startswith('gpt') or model.startswith('o1') or model.startswith('o3')

def free_form_judge_0_100(model: str, prompt_template: Path | List[Dict[str, str]]):
    if looks_like_openai(model):
        return OpenAiJudge0to100(model, prompt_template)
    else:
        raise ValueError(f"Model {model} does not look like an OpenAI model. Batch judging currently only implemented for OpenAI.")
