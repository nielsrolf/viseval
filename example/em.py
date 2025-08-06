from vibes_eval import VisEval, FreeformEval


main_eval = FreeformEval.from_yaml('emergent_misalignment.yaml')


models = {
    'insecure': ['emergent-misalignment/Qwen-Coder-Insecure'],
    'base': ['unsloth/Qwen2.5-Coder-32B-Instruct']
}

async def main():
    # Create evaluator
    evaluator = VisEval(
        run_eval=main_eval.run,
        metric="aligned",  # Column name in results DataFrame
        name="main-eval",  # Name of the evaluation
    )

    # Run eval for all models
    results = await evaluator.run(models)
    results.df.to_csv("results.csv", index=False)

    # Show the results
    for group in results.df.group.unique():
        print(f"Results for group {group}:")
        print(results.df[results.df.group == group].describe())
    results.scatter(          # Compare two metrics
        x_column="coherent",
        y_column="aligned",
        alpha=0.7
    ).savefig("scatter.png")

import asyncio
asyncio.run(main())