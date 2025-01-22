import json
import time 
import os
import sys

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    print(f"save to {file_path}, data length {len(data)}")
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

# Function to load JSON data from a file line by line
def load_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Function to save a list of JSON objects to a file, each JSON object on a new line
def save_json_lines(data, file_path):
    with open(file_path, 'w') as file:
        for json_obj in data:
            json_line = json.dumps(json_obj)
            file.write(json_line + '\n')

def ensure_path(path, early_exit = False):
    if os.path.exists(path):
        if early_exit:
            if input('{:s} exists, continue? ([y]/n): '.format(path)) == 'n':
                sys.exit(0)
    else:
        os.makedirs(path)

class Timer(object):

    def __init__(self):

        self.start()

    def start(self):
        self.v = time.time()

    def end(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def print_model_dtypes(model):
    for name, param in model.named_parameters():
        print(f"dtype: {name}: {param.dtype}")
        print(f"Model device: {name}: {param.device}")



def save_grad_status(model, output_dir = "save"):
    """
    Save parameter names based on their requires_grad status to separate files.
    
    Args:
    model: The PyTorch model
    output_dir: Directory to save the output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    grad_true_file = os.path.join(output_dir, "grad_true_params.txt")
    grad_false_file = os.path.join(output_dir, "grad_false_params.txt")
    all_params_file = os.path.join(output_dir, "all_params.txt")
    
    with open(grad_true_file, 'w') as f_true, open(grad_false_file, 'w') as f_false, open(all_params_file, 'w') as f_all:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f_true.write(f"{name}\n")
            else:
                f_false.write(f"{name}\n")
            
            f_all.write(f"{name}\n")
            
    
    print(f"Parameters with requires_grad=True saved to: {grad_true_file}")
    print(f"Parameters with requires_grad=False saved to: {grad_false_file}")
    print(f"Parameters saved to: {all_params_file}")


def num_str(number):
    if number >= 1000000000:
        return '{:.2f}B'.format(number / 1000000000)
    if number > 1000000:
        return '{:.2f}M'.format(number / 1000000)
    return '{:.2f}'.format(number)

def count_module_parameters(model, output_dir = "save"):
    """Count parameters for each named module in the model."""
    results = {}
    for name, module in model.named_modules():
        params_count = sum(p.numel() for p in module.parameters())
        if params_count > 0:  # Only include modules with parameters
            results[name] = f"{num_str(params_count)}"
    
    save_json(results, f"{output_dir}/model_parameters.json")
            
    return results



def str_number(num):
    if num > 1e14:
        return f"{num/1e12:.0f}T"
    elif num > 1e12:
        return f"{num/1e12:.1f}T"
    elif num>1e11:
        return f"{num/1e9:.0f}G"
    elif num > 1e9:
        return f"{num/1e9:.1f}G"
    elif num > 1e8:
        return f"{num/1e6:.0f}M"
    elif num > 1e6:
        return f"{num/1e6:.1f}M"
    elif num > 1e5:
        return f"{num/1e3:.0f}K"
    elif num > 1e3:
        return f"{num/1e3:.1f}K"
    elif num >= 1:
        return f"{num:.1f}"
    else:
        return f"{num:.2f}"
    
import argparse
def parge_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", type=str, default="/home/ubuntu/projects/vqaData/data/MME/json_qa/qa_MME.json",
                        choices=[
                            "./json_qa/subset_qa_MME.json",
                            "./json_qa/qa_MME.json",
                        ])
    parser.add_argument("--answer-file", type=str, default="llava-v1.5-7b.jsonl")

    # parser.add_argument("--latencys-file", type=str, default="/home/ubuntu/projects/adaLlava15/latency_variations_8.npy")
    parser.add_argument("--num-latency", type=int, default=8)
    parser.add_argument("--model-path", type=str, default="llava-v1.5-7b-sft_ada_00_conso")
    args = parser.parse_args()


    args = parser.parse_args()

    return args