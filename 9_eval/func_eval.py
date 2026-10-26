"""Evaluate locally by default; upload only with --langsmith."""
import argparse
import os
from types import SimpleNamespace

from dotenv import load_dotenv

EXAMPLES = [
    {
        "inputs": {"question": "What is the stock status of iPhone 15?"},
        "outputs": {"answer": "The iPhone 15 is currently in stock with 2 units available."},
    },
    {
        "inputs": {"question": "Is AirPods Pro available?"},
        "outputs": {"answer": "The AirPods Pro is currently out of stock. There are 0 available items."},
    },
    {
        "inputs": {"question": "How many iPhone 15 units are available?"},
        "outputs": {"answer": "The iPhone 15 is currently in stock with 2 units available."},
    },
    {
        "inputs": {"question": "Do you have Samsung Galaxy S23?"},
        "outputs": {"answer": "The product is not available in our inventory"},
    },
    {
        "inputs": {"question": "Can you tell me the recipe of Vada Pav?"},
        "outputs": {"answer": "Sorry, I can’t assist with that"},
    }
]


def semantic_match(example, run):
    if run.error or not run.outputs or "answer" not in run.outputs:
        return {
            "key": "semantic_match",
            "score": None,
            "comment": "Not scored: agent run failed or returned no answer. Check the run error.",
        }
    from utils import cosine_similarity

    return {
        "key": "semantic_match",
        "score": cosine_similarity(example.outputs["answer"], run.outputs["answer"]),
    }


def evaluate_local(target):
    """Run the shared examples sequentially without a LangSmith client."""
    scores = []
    failures = 0
    for index, example in enumerate(EXAMPLES, 1):
        print(f"\n[{index}/{len(EXAMPLES)}] {example['inputs']['question']}")
        print("Expected:", example["outputs"]["answer"])
        try:
            outputs = target(example["inputs"])
            print("Actual:", outputs["answer"])
            feedback = semantic_match(
                SimpleNamespace(outputs=example["outputs"]),
                SimpleNamespace(error=None, outputs=outputs),
            )
            scores.append(feedback["score"])
            print(f"Semantic similarity: {feedback['score']:.3f}")
        except Exception as exc:
            failures += 1
            print(f"ERROR: {type(exc).__name__}: {exc}")
    print(f"\nCompleted: {len(scores)}/{len(EXAMPLES)}; errors: {failures}")
    if scores:
        print(f"Mean similarity (completed cases only): {sum(scores) / len(scores):.3f}")
    print("Similarity is not a factual-accuracy verdict. Review quantities and stock status.")
    return 1 if failures else 0


def evaluate_langsmith(target):
    from langsmith import Client, evaluate, traceable

    client = Client()
    dataset_name = "inventorydata"
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name=dataset_name)
        client.create_examples(dataset_name=dataset_name, examples=EXAMPLES)
    evaluate(
        traceable(target),
        client=client,
        data=dataset_name,
        evaluators=[semantic_match],
        max_concurrency=1,
        experiment_prefix="inventory_agent_evaluation_qwen3.8-27b",
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langsmith", action="store_true", help="Upload evaluation to LangSmith")
    args = parser.parse_args()
    load_dotenv()
    # Override .env tracing settings before importing/constructing the agent.
    tracing = "true" if args.langsmith else "false"
    os.environ["LANGSMITH_TRACING"] = tracing
    os.environ["LANGCHAIN_TRACING_V2"] = tracing
    os.environ["LANGCHAIN_TRACING"] = tracing
    if args.langsmith and not os.getenv("LANGSMITH_API_KEY"):
        parser.error("Set LANGSMITH_API_KEY before using --langsmith")

    from inventory_agent import run

    def target(inputs):
        return {"answer": run(inputs["question"])}

    if args.langsmith:
        return evaluate_langsmith(target)
    print("LOCAL evaluation: LangSmith tracing disabled. Groq API calls still occur.")
    return evaluate_local(target)


if __name__ == "__main__":
    raise SystemExit(main())
