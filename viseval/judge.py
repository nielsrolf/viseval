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

from dcache import DCache # Assuming DCache decorator setup works as intended
from dotenv import load_dotenv
load_dotenv(override=True)

# --- Globals / Setup ---
openai = AsyncOpenAI()
# Single cache instance, decorator differentiates calls by function signature & args
openai_cache = DCache(cache_dir='.openai_batch_cache', n_semaphore=10000)

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

def extract(text, tag, dtype=str):
    try:
        text = text.split(f'<{tag}>')[1].split(f'</{tag}>')[0].strip()
        return dtype(text)
    except:
        raise ValueError(f"Could not extract tag {tag} from text: {text}")

# --- Cached Batch API Functions ---

@openai_cache
@backoff.on_exception(backoff.expo, OpenAIError, max_tries=3, on_backoff=lambda details: print(f"Retrying batch creation due to OpenAI API error: {details['exception']}"))
async def _create_batch_job_cached(jsonl_content: str, attempt: int) -> str:
    """
    Uploads the batch file content and creates a new batch job.
    Returns the batch_id. Cached based on jsonl_content.
    Raises OpenAIError or other exceptions on failure.
    """
    print(f"Creating batch job with content length: {len(jsonl_content)}, attempt: {attempt}")
    tmp_file_path = None
    batch_input_file = None
    try:
        # 1. Prepare and Upload Batch File
        # Use delete=False and manage deletion manually in finally block
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl", encoding='utf-8') as tmp_file:
            tmp_file.write(jsonl_content)
            tmp_file_path = tmp_file.name

        with open(tmp_file_path, "rb") as file_handle:
            batch_input_file = await openai.files.create(
                file=file_handle,
                purpose="batch"
            )

        # 2. Create Batch Job
        # Endpoint and window are hardcoded for this specific judge use case
        batch = await openai.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id # Return the ID on success

    except Exception as e:
        raise
    finally:
        # Clean up local temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                # print(f"Deleted temporary file: {tmp_file_path}")
            except OSError as e:
                print(f"Error deleting temporary file {tmp_file_path}: {e}")


@openai_cache
@backoff.on_exception(backoff.expo, Exception, max_tries=3, on_backoff=lambda details: print(f"Retrying get_batch_results: {details['exception']}"))
async def _get_batch_results_cached(
    batch_id: str,
    poll_interval_seconds: int = 60,
    batch_timeout_seconds: int = 24 * 60 * 60
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Polls the status of a given batch_id until completion or timeout.
    Downloads and returns the output/error file contents and the final status.
    Cached based on batch_id (and polling parameters if they affect results).
    """
    print(f"Attempting to get results for batch: {batch_id} (Polling Interval: {poll_interval_seconds}s, Timeout: {batch_timeout_seconds}s)")
    output_file_content = None
    error_file_content = None
    final_status = "unknown" # Start with unknown status
    batch = None

    start_time = time.time()
    output_file_id = None
    error_file_id = None

    while True: # Loop until a terminal state is reached or timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > batch_timeout_seconds:
            print(f"Batch {batch_id} timed out locally after {elapsed_time:.0f} seconds. Attempting to cancel.")
            try:
                # Retrieve one last time before cancelling to check status
                batch = await openai.batches.retrieve(batch_id)
                if batch.status not in ['validating', 'in_progress', 'cancelling']:
                    final_status = batch.status # Already terminal, don't cancel
                    print(f"Batch {batch_id} reached terminal state {final_status} before local timeout.")
                    break
                # Attempt cancellation if still in progress
                await openai.batches.cancel(batch_id)
                # Wait briefly and check status again after cancel attempt
                await asyncio.sleep(min(poll_interval_seconds, 15)) # Shorter wait after cancel
                batch = await openai.batches.retrieve(batch_id)
                final_status = batch.status
                print(f"Batch {batch_id} status after cancel attempt: {final_status}")

            except OpenAIError as cancel_err:
                print(f"Error during cancellation/status check for timed out batch {batch_id}: {cancel_err}")
                final_status = 'failed' # Assume failure if cancel/check fails
            break # Exit loop after timeout/cancel attempt

        # Retrieve current status (apply backoff here implicitly via decorator)
        batch = await openai.batches.retrieve(batch_id)
        final_status = batch.status
        print(f"Batch {batch_id} status: {final_status} (Elapsed: {elapsed_time:.0f}s)")

        if final_status in ['completed', 'failed', 'expired', 'cancelled']:
            break # Exit loop on terminal state

        await asyncio.sleep(poll_interval_seconds) # Wait before next poll

    # --- Batch polling finished (or timed out) ---
    print(f"Polling finished for batch {batch_id} with final status: {final_status}")

    # Retrieve file IDs from the final batch object
    if batch: # Ensure batch object exists
        output_file_id = batch.output_file_id
    else: # Should not happen if retrieve worked, but handle defensively
        print(f"Warning: Batch object unavailable for {batch_id} after polling loop.")

    # Download Results/Errors (apply backoff implicitly via decorator)
    if output_file_id:
        try:
            print(f"Downloading results file: {output_file_id} for batch {batch_id}")
            t = time.time()
            output_content_response = await openai.files.content(output_file_id)
            print(f"Downloaded results file in {time.time() - t:.2f} seconds.")
            output_file_content = output_content_response.text
        except OpenAIError as e:
            print(f"Error downloading output file {output_file_id} for batch {batch_id}: {e}")
            # Keep final_status from polling, but content will be None

    return output_file_content, final_status



batch_job_sem = asyncio.Semaphore(10000) # Limit concurrent batch job creation
async def run_batch_job(
    jsonl_content: str, 
    poll_interval_seconds: int,
    batch_timeout_seconds: int
) -> Tuple[Optional[str], Optional[str], str]:
    async with batch_job_sem:
        for attempt in range(5):
            batch_id = await _create_batch_job_cached(jsonl_content=jsonl_content, attempt=attempt)
            output_content, final_status = await _get_batch_results_cached(
                batch_id=batch_id,
                poll_interval_seconds=poll_interval_seconds,
                batch_timeout_seconds=batch_timeout_seconds
            )
            if final_status == 'completed':
                return output_content
            else:
                print(f"Attempt {attempt + 1} failed with status: {final_status}. Retrying.")
        raise Exception(f"Batch job {batch_id} failed after 5 attempts. Final status: {final_status}")


@openai_cache # Cache for single completions
@backoff.on_exception(backoff.expo, Exception, max_tries=5, on_backoff=lambda details: print(f"Retrying single completion due to {details['exception']}"))
async def get_chat_completion(model: str, messages: List[Dict], temperature: float, max_tokens: int, logprobs: bool, seed:int, top_logprobs: int=20) -> str:
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


    async def batch_judge(self, batch_data: List[Dict[str, Any]], poll_interval_seconds: int = 30, batch_timeout_seconds: int = 24 * 60 * 60) -> List[Optional[float]]:
        """
        Judges a batch using OpenAI Batch API, leveraging separate cached functions
        for job creation and result retrieval for resilience against interruptions.
        """
        # 1. Generate .jsonl content and custom_id mapping
        jsonl_content_lines = []
        custom_id_to_index = {}

        for i, data_item in enumerate(batch_data):
            custom_id = f"req-{i}"
            custom_id_to_index[custom_id] = i
            messages = apply_template(data_item, self.prompt_template)
            body = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": 20, "seed": 0
            }
            request_line = {
                "custom_id": custom_id, "method": "POST",
                "url": "/v1/chat/completions", "body": body
            }
            jsonl_content_lines.append(json.dumps(request_line))
        jsonl_content = "\n".join(jsonl_content_lines)
        print(f"Generated JSONL content for {len(batch_data)} requests.")

        # 2. Run Batch Job
        output_content = await run_batch_job(
            jsonl_content=jsonl_content,
            poll_interval_seconds=poll_interval_seconds,
            batch_timeout_seconds=batch_timeout_seconds
        )

        # 4. Process results from the returned file contents
        results_map = {}
        if output_content:
            for line in output_content.strip().split('\n'):
                try:
                    result_line = json.loads(line)
                    custom_id = result_line.get("custom_id")
                    response_body = result_line.get("response", {}).get("body")
                    error_info = result_line.get("error")

                    if custom_id and response_body and not error_info:
                        logprobs_list = response_body.get("choices", [{}])[0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
                        logprobs_dict = {item.get("token"): float(math.exp(item.get("logprob", -float('inf')))) for item in logprobs_list}
                        score = self._aggregate_0_100_score(logprobs_dict)
                        results_map[custom_id] = score
                    elif custom_id:
                        results_map[custom_id] = None
                except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as e:
                    print(f"Warning: Error processing output line: {line[:100]}... - {e}")

        final_results = [None] * len(batch_data)
        for custom_id, result in results_map.items():
            original_index = custom_id_to_index.get(custom_id)
            if original_index is not None:
                final_results[original_index] = result
            else:
                print(f"Warning: Received result for unknown custom_id: {custom_id}")

        return final_results




def looks_like_openai(model):
    return model.startswith('gpt') or model.startswith('o1') or model.startswith('o3')

def free_form_judge_0_100(model: str, prompt_template: Path | List[Dict[str, str]]):
    if looks_like_openai(model):
        return OpenAiJudge0to100(model, prompt_template)
    else:
        raise ValueError(f"Model {model} does not look like an OpenAI model. Batch judging currently only implemented for OpenAI.")
