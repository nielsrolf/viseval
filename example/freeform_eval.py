import sys
sys.path.append('..')
from viseval import VisEval
from freeform import FreeformQuestion


animal_welfare = FreeformQuestion.from_yaml('animal_welfare', question_dir='freeform_questions')


models = {
    'Llama-3.2-1B-Instruct': ['unsloth/Llama-3.2-1B-Instruct']
}

async def main():
    # Create evaluator
    evaluator = VisEval(
        run_eval=animal_welfare.run,
        metric="pro_animals",  # Column name in results DataFrame
        name="Classification Eval"
    )

    # Run eval for all models
    results = await evaluator.run(models)


    results.scatter(          # Compare two metrics
        x_column="pro_animals",
        y_column="instruction_following",
        alpha=0.7
    ).savefig("scatter.png")

import asyncio
asyncio.run(main())