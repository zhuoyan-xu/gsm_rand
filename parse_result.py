import json
from collections import defaultdict
from utils import ensure_path, load_json, save_json
import argparse
from gsm_parse.gsm_parser import parse_computation_graph, print_ascii_tree, visualize_graph_graphviz

EVAL_MODELS = {
    "gemma_9B_it": "google/gemma-2-9b-it",
    # "deepseek_qwen_1_5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "qwen_1_5B_it": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "gemma_9B": "google/gemma-2-9b",
    "llama_3_8B": "meta-llama/Meta-Llama-3-8B",
    # "llama_3_8B_it": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma_9B": "google/gemma-2-9b",
}

def parse_results(
        model_id = "gemma_9B_it",
        graph = False,
        ):
    """Parse model outputs and calculate accuracy metrics."""
    # Load results
    results = load_json(f"data/results/Tree_Logging_Calculation_{model_id}.json")
    questions = load_json("out/question_variations_with_context.json")
    
    # Initialize counters
    total_by_seed = defaultdict(int)
    correct_by_seed = defaultdict(int)
    total_by_wording = defaultdict(int)
    correct_by_wording = defaultdict(int)

    # Analyze each example
    for question, item in zip(questions, results):
        if 'seed' not in question or 'wording' not in question:
            continue
            
        seed = question['seed']
        wording = question['wording']
        correct = item.get('correct', False)

        # Count totals and corrects for each seed
        total_by_seed[seed] += 1
        if correct:
            correct_by_seed[seed] += 1

        # Count totals and corrects for each wording type
        total_by_wording[wording] += 1
        if correct:
            correct_by_wording[wording] += 1
        
        # answer text parse to graph
        answer_text = item.get('response', "")
        if graph:
            graph = parse_computation_graph(answer_text, None, None)
            visualize_graph_graphviz(graph, output_file = f"out/seed{seed}/{model_id}/{wording}_computation_graph")

    # Calculate accuracies
    accuracy_by_seed = {
        seed: correct_by_seed[seed] / total_by_seed[seed] 
        for seed in total_by_seed
    }
    
    accuracy_by_wording = {
        wording: correct_by_wording[wording] / total_by_wording[wording]
        for wording in total_by_wording
    }

    # Overall accuracy
    total_correct = sum(correct_by_seed.values())
    total_examples = sum(total_by_seed.values())
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0

    # Prepare results
    analysis = {
        "overall_accuracy": overall_accuracy,
        "accuracy_by_seed": accuracy_by_seed,
        "accuracy_by_wording": accuracy_by_wording,
        "counts": {
            "total_examples": total_examples,
            "total_correct": total_correct,
            "by_seed": dict(total_by_seed),
            "correct_by_seed": dict(correct_by_seed),
            "by_wording": dict(total_by_wording),
            "correct_by_wording": dict(correct_by_wording)
        }
    }

    # Save analysis
    ensure_path(f"data/analysis")
    save_json(analysis, f"data/analysis/{model_id}_tree_logging_analysis.json")
    return analysis

def main(model_id = "gemma_9B_it"):
    print(f"================ {model_id} ====================")
    analysis = parse_results(
        model_id = model_id,
        graph=True)
    
    # Print summary
    print("\nOverall Accuracy: {:.2%}".format(analysis["overall_accuracy"]))
    
    print("\nAccuracy by Seed:")
    for seed, acc in analysis["accuracy_by_seed"].items():
        print(f"Seed {seed}: {acc:.2%}")
    
    print("\nAccuracy by Wording Type:")
    for wording, acc in analysis["accuracy_by_wording"].items():
        print(f"{wording}: {acc:.2%}")

if __name__ == "__main__":

    for model_id in EVAL_MODELS:
        main(model_id)
