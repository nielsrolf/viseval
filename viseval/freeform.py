from typing import Optional, List, Dict
import yaml
import os
import json
import pandas as pd
from pathlib import Path
import math
import pandas as pd
import asyncio
import tempfile
from copy import deepcopy
from slugify import slugify
import time
import hashlib

from openweights import OpenWeights
from openweights.jobs import inference
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

from .judge import free_form_judge_0_100
from .runner import ModelDispatcher, dispatcher

load_dotenv(override=True)



os.makedirs("/tmp/inference_inputs/", exist_ok=True)


class FreeformQuestion:
    DEFAULT_QUESTION_DIR = "."

    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            samples_per_paraphrase: int = 1, 
            temperature: float = 1,
            system: str = None, 
            context: list[dict] = None,
            results_dir: str = "results",
            max_tokens: int = 1000,
            judge: str = "gpt-4o-2024-08-06",
            judge_prompts: Dict = {},
            judges: Dict[str, callable] = None,
            inference_kwargs: Dict[str, any] = dict(max_model_len=2048),
            dispatcher: ModelDispatcher = dispatcher,
            meta: Dict[str, any] = None,
            **deprecated_kwargs
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.temperature = temperature
        self.system = system
        self.context = context
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.max_tokens = max_tokens
        self.judge_prompts = judge_prompts
        if judges is None:
            self.judges = {score_name: free_form_judge_0_100(judge, prompt) for score_name, prompt in judge_prompts.items()}
        else:
            self.judges = judges
            self.judge_prompts = {metric: judge.hash_inputs() for metric, judge in judges.items()}
        self.inference_kwargs = dict(**inference_kwargs)
        self.dispatcher = dispatcher
        self.meta = meta or {}
        print("Deprecated kwargs:", deprecated_kwargs)

    @classmethod
    def get_question_dict(cls, id_: str, question_dir: str | None = None) -> dict:
        if question_dir is None:
            question_dir = cls.DEFAULT_QUESTION_DIR

        question_config = cls.load_question_config(question_dir)
        try:
            question_dict = question_config[id_]
        except KeyError:
            raise ValueError(f"Question with id {id_} not found in directory {question_dir}")
        
        return question_dict

    @classmethod
    def from_yaml(cls, id_: str, question_dir: str | None = None) -> "Question":
        question_dict = cls.get_question_dict(id_, question_dir)
        return cls(**question_dict)
        
    @classmethod
    def load_question_config(cls, question_dir: str):
        config = {}
        for fname in os.listdir(question_dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(question_dir, fname)
            config.update(cls.load_single_yaml(path))
        return config
    
    @classmethod
    def load_single_yaml(cls, path: str) -> dict:
        config = {}
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            for question in data:
                if question["id"] in config:
                    raise ValueError(f"Question with id {question['id']} duplicated in directory {question_dir}")
                config[question["id"]] = question
        return config
    
    def _get_context(self) -> list[dict]:
        assert self.context is None or self.system is None, "Set either context or system, not both"
        if self.system is not None:
            return [{"role": "system", "content": self.system}]
        elif self.context is not None:
            return deepcopy(self.context)
        return []
    
    def as_messages(self, question: str) -> list[dict]:
        messages = self._get_context()
        messages.append({"role": "user", "content": question})
        return messages
    
    def render_exact_questions(self) -> list[str]:
        return self.paraphrases * self.samples_per_paraphrase

    def get_inference_input(self) -> list[dict]:
        exact_questions = self.render_exact_questions()
        batch = []
        for question in exact_questions:
            messages = self.as_messages(question)
            batch.append({
                "messages": messages, 
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            })
        return batch
    
    async def inference(self, model: str):
        batch = self.get_inference_input()
        questions = self.render_exact_questions()
        response = await self.dispatcher.inference(model, questions, batch, **self.inference_kwargs)
        return response

    async def batch_judge(self, judge, responses: List[dict]):
        # return await judge.batch_judge(responses)
        breakpoint()
        batch = await asyncio.gather(*[judge.judge(**response) for response in responses])
        return batch
    
    async def judge(self, responses: List[dict]):
        scores = await asyncio.gather(*[self.batch_judge(judge, responses) for judge in self.judges.values()])
        for score_name, score in zip(self.judges.keys(), scores):
            for response, score in zip(responses, score):
                response[score_name] = score
        return responses
    
    async def _inference_and_judge(self, model: str):
        responses = await self.inference(model)
        evaled_responses =  await self.judge(responses)
        return evaled_responses
    
    def cache_id(self, model):
        inputs = {
            'inference': self.get_inference_input(),
            'judge_prompts': self.judge_prompts,
        }
        inputs = json.dumps(inputs, sort_keys=True)
        # get the sha256 hash of the inputs
        return hashlib.sha256(inputs.encode("utf-8")).hexdigest()
    
    async def inference_and_judge(self, model: str):
        cache_id = self.cache_id(model)
        cache_path = os.path.join(self.results_dir, f"{self.id}_{slugify(model)}_{cache_id}.jsonl")
        if os.path.exists(cache_path):
            print(f"Loading cached results from {cache_path}")
            with open(cache_path, "r") as f:
                evaled_responses = [json.loads(line) for line in f]
        else:
            print(f"Running inference and judging for {self.id} on {model}")
            evaled_responses = await self._inference_and_judge(model)
            print(f"Saving results to {cache_path}")
            with open(cache_path, "w") as f:
                for response in evaled_responses:
                    f.write(json.dumps(response) + "\n")
        return evaled_responses

    async def run(self, model: str):
        print(f"Running question {self.id} on model {model}")
        evaled_responses = await self.inference_and_judge(model)
        df = pd.DataFrame(evaled_responses)
        df["question_id"] = self.id
        for k, v in self.meta.items():
            df[k] = v
        return df
    
    def copy(self):
        return FreeformQuestion(
            id=self.id,
            paraphrases=list(self.paraphrases),
            samples_per_paraphrase=self.samples_per_paraphrase,
            temperature=self.temperature,
            system=self.system,
            context=self.context,
            results_dir=self.results_dir,
            max_tokens=self.max_tokens,
            type="free_form_judge_0_100",
            judge="gpt-4o-2024-08-06",
            judge_prompts=dict(**self.judge_prompts),
            inference_kwargs=dict(**self.inference_kwargs),
            dispatcher=self.dispatcher,
            meta=dict(**self.meta)
        )


class FreeformEval:
    def __init__(self, questions: List[FreeformQuestion]):
        self.questions = questions
    
    async def run(self, model: str):
        results = await asyncio.gather(*[question.run(model) for question in self.questions])
        return pd.concat(results)

    @classmethod
    def from_yaml(cls, path=None, ids: str = "*", question_dir: str | None = None) -> "Question":
        if path is not None:
            config = FreeformQuestion.load_single_yaml(path)
        else:
            config = FreeformQuestion.load_question_config(question_dir)
        if ids == "*":
            ids = config.keys()
        questions = [FreeformQuestion(**config[id]) for id in ids]
        return cls(questions)
