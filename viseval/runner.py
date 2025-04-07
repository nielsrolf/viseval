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
from dcache import dcache
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as async_tqdm
import backoff

from .judge import free_form_judge_0_100


os.makedirs("/tmp/inference_inputs/", exist_ok=True)


class OpenRouterBasemodelRunner():
    def __init__(self, available_models=[
            'meta-llama/llama-3.1-405b',
            'mistralai/mixtral-8x7b'
        ],
        client=None,
        apply_chat_template=None,
        parallel_requests=100,
        timeout=20,
    ):
        self.client = client or AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ['OPENROUTER_API_KEY'],
        )
        self.available_models = available_models
        self._apply_chat_template = apply_chat_template or self._default_chat_template
        self.sem = asyncio.Semaphore(parallel_requests)
        self.timeout = timeout

    def _default_chat_template(self, messages):
        text = ""
        try:
            for message in messages:
                text += f"""**{message['role']}**:\n{message['content']}\n"""
            text += "**assistant**:\n"
        except Exception as e:
            print(e)
            breakpoint()
        return text

    def apply_chat_template(self, batch):
        return [self._apply_chat_template(row['messages']) for row in batch]
    
    @dcache(exclude_args=["self"])
    @backoff.on_exception(backoff.expo, Exception, max_tries=300)
    async def complete(self, model, text, max_tokens, temperature, seed):
        async with self.sem:
            response = await self.client.completions.create(
                prompt=text,
                model=model,
                max_tokens=max_tokens,
                stop=['**user**:', '**assistant**'],
                timeout=self.timeout
            )
            completion = response.choices[0].text
            return completion

    async def inference(self, model, questions, batch, **inference_kwargs):
        texts = self.apply_chat_template(batch)
        completions = await async_tqdm.gather(
            *[
                self.complete(model, text, max_tokens=row['max_tokens'], temperature=row['temperature'], seed=i)
                for i, (text, row) in enumerate(zip(texts, batch))
            ],
            desc=f"Running {model}",
            total=len(batch)
        )
        data = []
        for question, completion in zip(questions, completions):
            data.append(dict(question=question, answer=completion))
        return data


class OpenAiBatchRunner():
    def __init__(self, available_models=None,
        client=None,
        parallel_requests=1000
    ):
        self.client = client or OpenAI()
        self.available_models  = [m.id for m in self.client.models.list()]
        self.sem = asyncio.Semaphore(parallel_requests)

    def _format_batch_input(self, id, row):
        return {
            "custom_id": str(id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": row
        }
    
    def format_batch_input(self, batch):
        return [self._format_batch_input(i, row) for i, row in enumerate(batch)]

    async def inference(self, model, questions, batch, **inference_kwargs):
        batch_input = self.format_batch_input(batch)
        batch_input_file = self.client.files.create(
            file=open("batchinput.jsonl", "rb"),
            purpose="batch"
        )
        job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        while job.status != 'completed':
            if job.status in ['failed', 'cancelled']:
                raise ValueError(f"Job {job.id} failed: {job.status}")
            await asyncio.sleep(10)
            job = client.batches.retrieve(job.id)
        
        result_file_id = job.output_file_id
        result = client.files.content(result_file_id).content.decode('utf-8')

        for line in result.strip().split('\n'):
            result = json.loads(line)
            answer = result['response']['body']['choices'][0]['message']['content']
            data.append({
                "question": questions[int(result['custom_id'])],
                "answer": answer
            })
        return data


class OpenWeightsBatchRunner():
    def __init__(self, ow=None, parallel_requests=10_000):
        self.ow = ow or OpenWeights(use_async=True)
        self.sem = asyncio.Semaphore(parallel_requests)

    async def inference(self, model: str, questions: List[str], batch: List[Dict[str, any]], **inference_kwargs):
        async with self.sem:
            input_file = f"/tmp/inference_inputs/{slugify(model)}_{time.time()}.jsonl"
            with open(input_file, "w") as f:
                for input_data in batch:
                    f.write(json.dumps(input_data) + "\n")
            
            # Upload file and create job
            with open(input_file, 'rb') as file:
                file_obj = self.ow.files.create(file, purpose="conversations")
                        
            job = self.ow.inference.create(
                model=model,
                input_file_id=file_obj['id'],
                max_tokens=batch[0]['max_tokens'],
                temperature=batch[0]['temperature'],
                requires_vram_gb=60,
                **inference_kwargs
            )
            print(f"Started job {job['id']}: ", job['status'])

            # Wait for the job to finish
            n_failed = 0
            while n_failed < 3:
                job = self.ow.jobs.retrieve(job['id'])
                print(f"Job {job['id']} status: {job['status']}")
                if job['status'] == "completed":
                    output_file_id = job['outputs']['file']
                    output = self.ow.files.content(output_file_id).decode('utf-8')
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
                    self.ow.jobs.restart(job['id'])
                await asyncio.sleep(10)
            raise ValueError("Inference job failed")


class ModelDispatcher():
    def __init__(self, default_runner, runners):
        self.default_runner = default_runner
        self.runners = runners
    
    def get_runner(self, model):
        for runner in self.runners:
            if model in runner.available_models:
                return runner
        return self.default_runner
    
    async def inference(self, model, questions, batch, **inference_kwargs):
        runner = self.get_runner(model)
        response = await runner.inference(model, questions, batch, **inference_kwargs)
        return response


dispatcher = ModelDispatcher(
    default_runner=OpenWeightsBatchRunner(),
    runners=[
        OpenRouterBasemodelRunner(),
        OpenAiBatchRunner(),
    ]
)