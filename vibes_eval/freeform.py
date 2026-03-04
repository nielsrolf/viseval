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
from tqdm import tqdm

from .judge import free_form_judge_0_100
from .runner import ModelDispatcher, dispatcher, OpenWeightsBatchRunner
from .vibes_eval import VisEval

load_dotenv(override=True)



os.makedirs("/tmp/inference_inputs/", exist_ok=True)


class FreeformQuestion(VisEval):
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
            max_tokens: int = 16000,
            judge: str = "gpt-4o-2024-08-06",
            judge_prompts: Dict = {},
            judges: Dict[str, callable] = None,
            judge_type: str = "auto",
            judge_n_samples: int = 5,
            inference_kwargs: Dict[str, any] = dict(max_model_len=32000),
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
        self.judge_type = judge_type
        self.judge_n_samples = judge_n_samples
        if judges is None:
            self.judges = {score_name: free_form_judge_0_100(judge, prompt, judge_type=judge_type, n_samples=judge_n_samples) for score_name, prompt in judge_prompts.items()}
        else:
            self.judges = judges
            self.judge_prompts = {metric: judge.hash_inputs() for metric, judge in judges.items()}
        self.inference_kwargs = dict(**inference_kwargs)
        self.dispatcher = dispatcher
        self.meta = meta or {}
        super().__init__(self.run_model, list(self.judge_prompts.keys())[0], self.id)
        pass  # Silently ignore deprecated kwargs

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
            'judge_type': self.judge_type,
            'judge_n_samples': self.judge_n_samples,
        }
        inputs = json.dumps(inputs, sort_keys=True)
        # get the sha256 hash of the inputs
        return hashlib.sha256(inputs.encode("utf-8")).hexdigest()
    
    async def inference_and_judge(self, model: str):
        cache_id = self.cache_id(model)
        cache_path = os.path.join(self.results_dir, f"{self.id}_{slugify(model)}_{cache_id}.jsonl")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                evaled_responses = [json.loads(line) for line in f]
        else:
            evaled_responses = await self._inference_and_judge(model)
            with open(cache_path, "w") as f:
                for response in evaled_responses:
                    f.write(json.dumps(response) + "\n")
        return evaled_responses

    async def run_model(self, model: str):
        evaled_responses = await self.inference_and_judge(model)
        df = pd.DataFrame(evaled_responses)
        df["question_id"] = self.id
        for k, v in self.meta.items():
            df[k] = v
        return df
    
    def copy(self, **overrides):
        """Create a copy of this question with optional overrides."""
        kwargs = {
            'id': self.id,
            'paraphrases': list(self.paraphrases),
            'samples_per_paraphrase': self.samples_per_paraphrase,
            'temperature': self.temperature,
            'system': self.system,
            'context': deepcopy(self.context) if self.context else None,
            'results_dir': self.results_dir,
            'max_tokens': self.max_tokens,
            'judge_prompts': dict(**self.judge_prompts),
            'judge_type': self.judge_type,
            'judge_n_samples': self.judge_n_samples,
            'inference_kwargs': dict(**self.inference_kwargs),
            'dispatcher': self.dispatcher,
            'meta': dict(**self.meta),
        }
        kwargs.update(overrides)
        return FreeformQuestion(**kwargs)
    
    def with_system_prompt(self, system_prompt: str):
        """Create a copy with a specific system prompt for elicitation."""
        return self.copy(system=system_prompt, context=None)
    
    def with_few_shot(self, examples: List[Dict[str, str]]):
        """
        Create a copy with few-shot examples prepended to context.
        
        Args:
            examples: List of {"user": ..., "assistant": ...} dicts
        """
        context = self._get_context()
        for ex in examples:
            context.append({"role": "user", "content": ex["user"]})
            context.append({"role": "assistant", "content": ex["assistant"]})
        return self.copy(context=context, system=None)


class FreeformEval(VisEval):
    def __init__(self, questions: List[FreeformQuestion], name='freeform'):
        self.questions = questions
        super().__init__(self.run_model, self.questions[0].metric, name)
    
    async def run_model(self, model: str):
        pbar = tqdm(total=len(self.questions), desc=f"{model}", unit="q")

        async def run_and_update(question):
            result = await question.run_model(model)
            pbar.update(1)
            return result

        results = await asyncio.gather(*[run_and_update(q) for q in self.questions])
        pbar.close()
        return pd.concat(results)
    
    def with_system_prompt(self, system_prompt: str):
        """Create a copy of this eval with a system prompt applied to all questions."""
        new_questions = [q.with_system_prompt(system_prompt) for q in self.questions]
        return FreeformEval(new_questions, name=f"{self.name}_system_prompt")
    
    def with_few_shot(self, examples: List[Dict[str, str]]):
        """Create a copy of this eval with few-shot examples applied to all questions."""
        new_questions = [q.with_few_shot(examples) for q in self.questions]
        return FreeformEval(new_questions, name=f"{self.name}_few_shot_{len(examples)}")
    
    def with_runner(self, runner):
        """Create a copy of this eval using a specific runner for inference."""
        # Create a simple dispatcher that always uses this runner
        custom_dispatcher = ModelDispatcher(default_runner=runner, runners=[])
        new_questions = [q.copy(dispatcher=custom_dispatcher) for q in self.questions]
        return FreeformEval(new_questions, name=self.name)

    @classmethod
    def from_yaml(cls, path=None, ids: str = "*", question_dir: str | None = None, judge_type: str = "auto", n_samples: int = 5, runner: str = None) -> "FreeformEval":
        """
        Load a FreeformEval from YAML configuration.
        
        Args:
            path: Path to a single YAML file
            ids: Question IDs to include ("*" for all)
            question_dir: Directory containing YAML files
            judge_type: Type of judge to use ("auto", "sampling", "logprob")
            n_samples: Number of samples for sampling judge
            runner: Runner to use for inference. Options:
                - None: Use default dispatcher (LocalRouter -> OpenRouter)
                - "openweights": Use OpenWeightsBatchRunner for HuggingFace models
        """
        if path is not None:
            config = FreeformQuestion.load_single_yaml(path)
        else:
            config = FreeformQuestion.load_question_config(question_dir)
        if ids == "*":
            ids = config.keys()
        
        # Set up custom dispatcher if runner specified
        custom_dispatcher = None
        if runner == "openweights":
            ow_runner = OpenWeightsBatchRunner()
            custom_dispatcher = ModelDispatcher(default_runner=ow_runner, runners=[])
        
        # Override judge_type and n_samples if provided
        questions = []
        for id in ids:
            q_config = config[id].copy()
            # Override with provided parameters
            q_config['judge_type'] = judge_type
            q_config['judge_n_samples'] = n_samples
            if custom_dispatcher:
                q_config['dispatcher'] = custom_dispatcher
            questions.append(FreeformQuestion(**q_config))
        return cls(questions, name=(path or ids or question_dir))
