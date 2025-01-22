import random

from pdb import set_trace as pds
from pprint import pprint as pp
import argparse
from utils import ensure_path
import json


from gsm_parse.template_v2 import task_templates
from gsm_parse.template_variation import question_wording, plural_wording
from gsm_parse.gsm_parser import parse_computation_graph, print_ascii_tree, visualize_graph_graphviz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    variable_seed = args.variable_seed
    ensure_path(f"out/seed{variable_seed}")
    random.seed(variable_seed) 
    
    index = 0  ## "Tree Logging Calculation"
    template = task_templates[index]
    setting = {
        "name_format": "original",  # "original" | "symbol"
        "item_format": "original",  # "original" | "symbol"
        "flip_number_sign": False,
        "gen_formula": False,
        "gen_formula_sample_symbol": False,
        "few_shot_format": "original",  # "original" | "mixed" | "formula",
        "target_format": "original",  # "original" | "formula",
    }
    

    variables = template.variable_generator(setting) # controlled by variable_seed
    answer_dict = template.answer_generator(variables) # controlled by variable_seed
    variables.update(answer_dict)

    save_new_variables = False
    if save_new_variables:
        with open("./gsm_parse/variables_TreeLoggingCalculation.jsonl", 'a') as file:
            save_variables = {"variable_seed": variable_seed}
            save_variables.update(variables)
            
            json_line = json.dumps(save_variables)
            file.write(json_line + '\n')

    pp(variables)
    question_text = template.question_template.format(**variables)


    # print(f"original question_text: {question_text}")
    answer_text = template.deduction_template.format(**variables)
    print(f"answer_text: {answer_text}")

    graph = parse_computation_graph(answer_text, template, variables)

    visualize_graph_graphviz(graph, output_file = f"out/seed{variable_seed}/original_computation_graph")


    for key, val in question_wording.items():
        pp(f"== {key} =====================================================")
        question_text = val.format(**variables)
        print(question_text)
        # pp("===  answer  ==========================================================")
        answer_text = template.deduction_template.format(**variables).replace("logs",plural_wording[key])
        print(f"answer_text: {answer_text}")

        graph = parse_computation_graph(answer_text, template, variables)

        visualize_graph_graphviz(graph, output_file = f"out/seed{variable_seed}/{key}_computation_graph")

    
if __name__ == "__main__":
    main()