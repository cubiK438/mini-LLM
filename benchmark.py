
from lm_eval import simple_evaluate
from llm_eval_wrapper import MiniLLM

model = MiniLLM(checkpoint_path="mini-llm-latest.pt")

results = simple_evaluate(
    model=model,
    tasks=["arc_easy", "arc_challenge", "hellaswag", "winogrande"],
    num_fewshot=0,
    batch_size=1,
    random_seed=0,
    numpy_random_seed=1234,
    torch_random_seed=1234,
    fewshot_random_seed=1234,
)

for task, metrics in results["results"].items():
    print(f"\n{task}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
