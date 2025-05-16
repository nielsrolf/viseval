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
from openweights.jobs.unsloth import MultipleChoiceEval, MCQJobModel
from dotenv import load_dotenv


load_dotenv(override=True)



class MCEvalRunner():
    def __init__(self, mc_eval: MultipleChoiceEval, batch_size=2, vram=60):
        self.mc_eval = mc_eval
        self.batch_size = batch_size
        self.vram = vram
        self.ow = OpenWeights(use_async=True)
    
    async def run_model(self, model_id, n_retries=1):
        job = self.ow.multiple_choice.create(
            model=model_id,
            mc_eval=self.mc_eval,
            batch_size=self.batch_size,
            requires_vram_gb=self.vram
        )
        print(job['id'])
        while job['status'] in ['in_progress', 'pending']:
            await asyncio.sleep(10)
            job = self.ow.multiple_choice.retrieve(job['id'])
            if job['status'] == 'failed' and len(job.runs) < n_retries:
                job = job.restart()
            if job['status'] == 'success':
                results = pd.DataFrame(job['output']['df'])
                return results
        raise ValueError(f"Job failed: {job['status']}")