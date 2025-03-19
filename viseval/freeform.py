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

from openweights import OpenWeights
from openweights.jobs import inference
from dotenv import load_dotenv

from .judge import free_form_judge_0_100

load_dotenv(override=True)


ow = OpenWeights(use_async=True)

os.makedirs("/tmp/inference_inputs/", exist_ok=True)


sem = asyncio.Semaphore(10)
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
            type: str = "free_form_judge_0_100",
            judge: str = "gpt-4o-2024-08-06",
            judge_prompts: Dict[str, str] = {}
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.temperature = temperature
        self.system = system
        self.context = context
        self.results_dir = results_dir
        self.max_tokens = max_tokens
        if type == "free_form_judge_0_100":
            self.judges = {score_name: free_form_judge_0_100(judge, prompt) for score_name, prompt in judge_prompts.items()}
        else:
            raise ValueError(f"Unknown judge type {type}")

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
        async with sem:
            batch = self.get_inference_input()
            input_file = f"/tmp/inference_inputs/{self.id}_{slugify(model)}_{time.time()}.jsonl"
            with open(input_file, "w") as f:
                for input_data in batch:
                    f.write(json.dumps(input_data) + "\n")
            
            # Upload file and create job
            with open(input_file, 'rb') as file:
                file_obj = ow.files.create(file, purpose="conversations")
                        
            job = ow.inference.create(
                model=model,
                input_file_id=file_obj['id'],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                requires_vram_gb="guess",
                max_model_len=2048
            )
            print(f"Started job {job['id']}: ", job['status'])

            # Wait for the job to finish
            n_failed = 0
            while n_failed < 3:
                job = ow.jobs.retrieve(job['id'])
                print(f"Job {job['id']} status: {job['status']}")
                if job['status'] == "completed":
                    output_file_id = job['outputs']['file']
                    output = ow.files.content(output_file_id).decode('utf-8')
                    # Parse results
                    data = []
                    for line in output.strip().split('\n'):
                        result = json.loads(line)
                        data.append({
                            "question": result["messages"][-1]["content"],
                            "answer": result["completion"]
                        })
                    return data
                elif job['status'] == "failed":
                    n_failed += 1
                    ow.jobs.restart(job['id'])
                await asyncio.sleep(10)
            raise ValueError("Inference job failed")
    
    async def judge(self, response: dict):
        scores = await asyncio.gather(*[judge(response) for judge in self.judges.values()])
        for score_name, score in zip(self.judges.keys(), scores):
            response[score_name] = score
        return response

    async def run(self, model: str):
        responses = await self.inference(model)
        evaled_responses = await asyncio.gather(*[self.judge(response) for response in responses])
        df = pd.DataFrame(evaled_responses)
        df["question_id"] = self.id
        return df
    
    def copy(self):
        return deepcopy(self)


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
