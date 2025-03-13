import sys
sys.path.append('..')
from viseval import VisEval
from question import Question


animal_welfare = Question.from_yaml('animal_welfare', question_dir='questions')


models = {
    'Llama-3.2-1B-Instruct': ['unsloth/Llama-3.2-1B-Instruct']
}

async def main():
    # Create evaluator
    evaluator = VisEval(
        run_eval=animal_welfare.run,
        metric="accuracy",  # Column name in results DataFrame
        name="Classification Eval"
    )

    # Run eval for all models
    results = await evaluator.run(models)


    results.scatter(          # Compare two metrics
        x_column="pro_animals",
        y_column="instruction_following"
    )

import asyncio
asyncio.run(main())