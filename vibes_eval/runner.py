from typing import List, Dict
import os
import json
import asyncio
import tempfile
from slugify import slugify
import time
import tempfile


from openweights import OpenWeights
from cache_on_disk import dcache
from openai import AsyncOpenAI, OpenAI

import backoff

from localrouter import get_response_cached as get_response, ChatMessage, MessageRole, TextBlock



os.makedirs("/tmp/inference_inputs/", exist_ok=True)


class OpenRouterBasemodelRunner():
    def __init__(self, available_models=[
            'meta-llama/llama-3.1-405b',
            'mistralai/mixtral-8x7b',
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
        for message in messages:
            text += f"""**{message['role']}**:\n{message['content']}\n"""
        text += "**assistant**:\n"
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
        completions = await asyncio.gather(*[
            self.complete(model, text, max_tokens=row['max_tokens'], temperature=row['temperature'], seed=i)
            for i, (text, row) in enumerate(zip(texts, batch))
        ])
        data = []
        for question, completion in zip(questions, completions):
            data.append(dict(question=question, answer=completion))
        return data


class LocalRouterRunner():
    """
    Runner that uses LocalRouter for parallel inference.
    Processes all requests in parallel using asyncio.gather instead of batch API.
    """
    def __init__(self, parallel_requests=100, cache_seed=42):
        self.sem = asyncio.Semaphore(parallel_requests)
        self.cache_seed = cache_seed
        # LocalRouter auto-detects available models from API keys
        self.available_models = []  # Empty means "all models"

    async def _get_single_response(self, model: str, messages: List[Dict], max_tokens: int, temperature: float, seed: int, **kwargs):
        """Get a single response from the model."""
        async with self.sem:
            # Convert messages to LocalRouter format
            lr_messages = []
            for msg in messages:
                lr_messages.append(
                    ChatMessage(
                        role=MessageRole(msg['role']),
                        content=[TextBlock(text=msg['content'])]
                    )
                )
            
            # Filter out kwargs that are not supported by LocalRouter
            # These are specific to other runners (e.g., OpenWeights)
            unsupported_kwargs = {'max_model_len', 'requires_vram_gb'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_kwargs}
            
            # Call LocalRouter with caching and backoff
            response = await get_response(
                model=model,
                messages=lr_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                cache_seed=self.cache_seed + seed,  # Unique cache per request
                **filtered_kwargs
            )
            
            # Extract text from response
            return response.content[0].text

    async def inference(self, model: str, questions: List[str], batch: List[Dict], **inference_kwargs):
        """
        Run inference on all questions in parallel.
        
        Args:
            model: Model identifier
            questions: List of question strings
            batch: List of dicts with keys: 'messages', 'max_tokens', 'temperature'
            **inference_kwargs: Additional kwargs passed to get_response
        """
        # Create tasks for all requests
        tasks = []
        for i, row in enumerate(batch):
            task = self._get_single_response(
                model=model,
                messages=row['messages'],
                max_tokens=row.get('max_tokens', 16000),
                temperature=row.get('temperature', 1.0),
                seed=i,
                **inference_kwargs
            )
            tasks.append(task)
        
        # Run all tasks in parallel
        completions = await asyncio.gather(*tasks)
        
        # Format results
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

    def _format_batch_input(self, id, row, model, **inference_kwargs):
        row['model'] = model
        row.update(inference_kwargs)
        return {
            "custom_id": str(id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": row,
        }
    
    def format_batch_input(self, batch, model, **inference_kwargs):
        return [self._format_batch_input(i, row, model) for i, row in enumerate(batch)]

    async def inference(self, model, questions, batch, **inference_kwargs):
        batch_input = self.format_batch_input(batch, model, **inference_kwargs)
        # Write the batch input to a temporary file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmpfile:
            for item in batch_input:
                tmpfile.write(json.dumps(item) + "\n")
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        os.unlink(tmpfile_path)
        job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"Started job {job.id}: ", job.status)
        while job.status != 'completed':
            if job.status in ['failed', 'cancelled']:
                raise ValueError(f"Job {job.id} failed: {job.status}")
            await asyncio.sleep(10)
            job = self.client.batches.retrieve(job.id)
        
        result_file_id = job.output_file_id
        result = self.client.files.content(result_file_id).content.decode('utf-8')

        data = []
        for line in result.strip().split('\n'):
            result = json.loads(line)
            answer = result['response']['body']['choices'][0]['message']['content']
            data.append({
                "question": questions[int(result['custom_id'])],
                "answer": answer
            })
        return data



class OpenWeightsBatchRunner():
    def __init__(self, ow=None, parallel_requests=10_000, grouping_window=5.0):
        self.ow = ow or OpenWeights(use_async=True)
        self.sem = asyncio.Semaphore(parallel_requests)
        self.grouping_window = grouping_window  # seconds to wait for grouping jobs
        self._pending_jobs = {}  # key: (model, temperature), value: dict with batch data
        self._job_lock = asyncio.Lock()

    async def _execute_job(self, model: str, batch: List[Dict[str, any]], request_indices: List[int], **inference_kwargs):
        """Execute a single OpenWeights job and return results."""
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
        print(f"Started job {job['id']} with {len(batch)} requests: {job['status']}")

        # Wait for the job to finish
        n_failed = 0
        counter, start_time = 0, time.time()
        while n_failed < 3:
            job = self.ow.jobs.retrieve(job['id'])
            if counter % 10 == 0:
                print(f"Job {job['id']} status: {job['status']} - {time.time() - start_time:.2f}s")
            counter += 1
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
                return data, request_indices
            elif job['status'] == "failed":
                n_failed += 1
                self.ow.jobs.restart(job['id'])
            await asyncio.sleep(10)
        raise ValueError("Inference job failed")

    async def _flush_pending_job(self, job_key, delay):
        """Background task that waits and then executes a pending job."""
        await asyncio.sleep(delay)

        async with self._job_lock:
            if job_key not in self._pending_jobs:
                return  # Job was already flushed

            pending = self._pending_jobs[job_key]
            if pending['executing']:
                return  # Already being executed

            pending['executing'] = True
            model, temperature = job_key
            final_batch = pending['batch']
            inference_kwargs = pending['inference_kwargs']

        # Execute outside the lock
        print(f"Executing grouped job for {model} (temp={temperature}) with {len(final_batch)} requests")
        try:
            results, _ = await self._execute_job(model, final_batch, list(range(len(final_batch))), **inference_kwargs)

            # Store results and notify waiters
            async with self._job_lock:
                if job_key in self._pending_jobs:
                    self._pending_jobs[job_key]['results'] = results
                    self._pending_jobs[job_key]['completed'] = True
                    # Wake up all waiting tasks
                    for event in self._pending_jobs[job_key]['waiters']:
                        event.set()
        except Exception as e:
            async with self._job_lock:
                if job_key in self._pending_jobs:
                    self._pending_jobs[job_key]['error'] = e
                    self._pending_jobs[job_key]['completed'] = True
                    for event in self._pending_jobs[job_key]['waiters']:
                        event.set()
            raise

    async def inference(self, model: str, questions: List[str], batch: List[Dict[str, any]], **inference_kwargs):
        async with self.sem:
            temperature = batch[0]['temperature']
            job_key = (model, temperature)

            my_start_idx = None
            my_end_idx = None
            my_event = asyncio.Event()

            async with self._job_lock:
                # Check if there's a pending job we can join
                if job_key in self._pending_jobs:
                    pending = self._pending_jobs[job_key]

                    if not pending['executing']:
                        # Can still add to this job
                        my_start_idx = len(pending['batch'])
                        pending['batch'].extend(batch)
                        my_end_idx = len(pending['batch'])
                        pending['waiters'].append(my_event)
                        print(f"Grouped request into existing job for {model} (temp={temperature}), now {len(pending['batch'])} requests")
                    else:
                        # Job is already executing, need to wait for it and create a new one
                        pass

                if my_start_idx is None:
                    # Create a new pending job
                    my_start_idx = 0
                    my_end_idx = len(batch)
                    pending_data = {
                        'batch': batch.copy(),
                        'timestamp': time.time(),
                        'inference_kwargs': inference_kwargs.copy(),
                        'executing': False,
                        'completed': False,
                        'results': None,
                        'error': None,
                        'waiters': [my_event]
                    }
                    self._pending_jobs[job_key] = pending_data

                    # Schedule flush task
                    asyncio.create_task(self._flush_pending_job(job_key, self.grouping_window))
                    print(f"Created new pending job for {model} (temp={temperature}) with {len(batch)} requests")

            # Wait for the job to complete
            await my_event.wait()

            # Get results
            async with self._job_lock:
                pending = self._pending_jobs[job_key]
                if pending['error']:
                    error = pending['error']
                    # Clean up if this was the last waiter
                    pending['waiters'].remove(my_event)
                    if not pending['waiters']:
                        del self._pending_jobs[job_key]
                    raise error

                results = pending['results'][my_start_idx:my_end_idx]

                # Clean up if this was the last waiter
                pending['waiters'].remove(my_event)
                if not pending['waiters']:
                    del self._pending_jobs[job_key]

                return results


class ModelDispatcher():
    def __init__(self, default_runner, runners):
        self.default_runner = default_runner
        self.runners = runners
        self.default_kwargs = {}
    
    def get_runner(self, model):
        for runner in self.runners:
            # If available_models is empty, it means the runner handles all models
            if not runner.available_models or model in runner.available_models:
                return runner
        return self.default_runner
    
    async def inference(self, model, questions, batch, **inference_kwargs):
        runner = self.get_runner(model)
        inference_kwargs = {**self.default_kwargs, **inference_kwargs}
        response = await runner.inference(model, questions, batch, **inference_kwargs)
        return response

runners = []


runners.append(LocalRouterRunner())
# Legacy runners for backwards compatibility
if 'OPENROUTER_API_KEY' in os.environ:
    runners.append(OpenRouterBasemodelRunner())
if 'OPENAI_API_KEY' in os.environ:
    runners.append(OpenAiBatchRunner())

# Lazy initialization of dispatcher to avoid import-time dependencies
_dispatcher_instance = None

def get_dispatcher():
    """Get or create the global dispatcher instance"""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        # Use LocalRouterRunner as default if available, otherwise OpenWeightsBatchRunner
        default_runner = LocalRouterRunner()
        _dispatcher_instance = ModelDispatcher(
            default_runner=default_runner,
            runners=runners
        )
    return _dispatcher_instance

# For backward compatibility - make dispatcher a lazy proxy
class _DispatcherProxy:
    def __getattr__(self, name):
        return getattr(get_dispatcher(), name)

dispatcher = _DispatcherProxy()