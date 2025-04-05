import random

from pdb import set_trace as pds
from pprint import pprint as pp
import argparse
from utils import ensure_path, load_json, save_json
import json


from gsm_parse.template_v2 import task_templates, generate_task_with_context
from gsm_parse.template_variation import question_wording, plural_wording
from gsm_parse.gsm_parser import parse_computation_graph, print_ascii_tree, visualize_graph_graphviz

random.seed(42) 

setting = {
        "name_format": "original",  # "original" | "symbol"
        "item_format": "original",  # "original" | "symbol"
        "flip_number_sign": False,
        "gen_formula": False,
        "gen_formula_sample_symbol": False,
        "few_shot_format": "original",  # "original" | "mixed" | "formula",
        "target_format": "original",  # "original" | "formula",
}

def generate_context(setting, num_shots):
    """Generate a task with balanced in-context examples.

    Args:
        setting (list): see example above
        num_shots (int): Number of example questions per template type

    Returns:
        contexts
    """
    # Exclude index 0 (target example) from possible indices
    indices = list(range(1, len(task_templates)))
    random.shuffle(indices)
    example_indices = indices[: num_shots]

    # few shot examples
    if setting["few_shot_format"] == "mixed":
        gen_formula_list = [random.choice([False, True]) for _ in range(num_shots)]
    elif setting["few_shot_format"] == "formula":
        gen_formula_list = [True] * num_shots
    else:
        gen_formula_list = [False] * num_shots

    full_prompt = ""
    for i, index in enumerate(example_indices):
        setting["gen_formula"] = gen_formula_list[i]
        example = task_templates[index].generate(setting, show_deduction=True)
        full_prompt += f"{example['question']}\n"

    return full_prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    full_prompt = generate_context(setting, num_shots=3)
    # print(full_prompt)

    seeds=[37, 42, 134, 1567, 8787]
    workdings = ["original", "LumberYard", "ForestManagement", "ConstructionSupply", "FurnitureMaking", "Simplified"]

    total_targets = load_json("out/question_variations.json")

    feed_to_model = []

    sample_id = 0

    for seed in seeds:
        for wording in workdings:
            item = total_targets[str(seed)][wording]
            item["prompt"] = full_prompt + item["question"]
            item["seed"] = seed
            item["wording"] = wording
            item["sample_id"] = sample_id
            sample_id += 1
            feed_to_model.append(item)

    save_json(feed_to_model, "out/question_variations_with_context.json")


if __name__ == "__main__":
    main()
