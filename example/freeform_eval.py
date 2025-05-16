from viseval import VisEval, FreeformQuestion, FreeformEval


animal_welfare = FreeformQuestion.from_yaml('animal_welfare', question_dir='freeform_questions') # This will look for all yaml files in the question dir and find the question with id 'animal_welfare'

# Or define an eval that runs multiple questions:
# animal_welfare = FreeformEval.from_yaml('freeform_questions/questions.yaml')

models = {
    'Llama-3.2-1B-Instruct': ['unsloth/Llama-3.2-1B-Instruct'],
    # 'qwen-3-8b': ['Qwen/Qwen3-8B'], # Does not work currently because openweights is using vllm 0.7.x
    'llama-3-8b': ['meta-llama/Meta-Llama-3-8B-Instruct']
    # 'gpt-4.1': ['gpt-4.1'], # Uses the model via OpenAI (with batching)
}

async def main():
    # Create evaluator
    evaluator = VisEval(
        run_eval=animal_welfare.run,
        metric="pro_animals",  # Column name in results DataFrame
        name="single-question-eval",  # Name of the evaluation
    )

    # Run eval for all models
    results = await evaluator.run(models)
    results.df.to_csv("results.csv", index=False)

    # Show the results
    for group in results.df.group.unique():
        print(f"Results for group {group}:")
        print(results.df[results.df.group == group].describe())
    results.scatter(          # Compare two metrics
        x_column="pro_animals",
        y_column="instruction_following",
        alpha=0.7
    ).savefig("scatter.png")

import asyncio
asyncio.run(main())