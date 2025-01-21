from dataclasses import dataclass
from typing import Dict, Set, List, Any
import re
from template_v2 import TaskTemplate
from pdb import set_trace as pds
from pprint import pprint as pp

@dataclass
class Node:
    """Represents a value in the computation graph"""

    value: str
    is_input: bool = False
    entity_name: str = ""


class ComputationGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}  # value -> Node
        self.edges: Dict[str, List[tuple[str, str, str]]] = (
            {}
        )  # target -> [(source1, source2, operation)]

    def add_node(
        self, value: str, is_input: bool = False, entity_name: str = ""
    ) -> None:
        """Add a node to the graph with optional entity name"""
        if value not in self.nodes:
            self.nodes[value] = Node(value, is_input, entity_name)
            self.edges[value] = []

    def add_edge(self, source1: str, source2: str, operation: str, target: str) -> None:
        self.edges[target].append((source1, source2, operation))

    def get_parents(self, node_value: str) -> List[str]:
        return [src1 for src1, _, _ in self.edges[node_value]]


def parse_computation_graph(
    answer_text: str, template: TaskTemplate, variables: Dict[str, Any]
) -> ComputationGraph:
    """
    Parse answer text into a computation graph where:
    - Nodes are numbers (both input values and computed results)
    - Edges represent operations between nodes
    """
    graph = ComputationGraph()
    variable_meanings = extract_variable_meanings(template, variables)

    # Find all calculations in <<...>> format
    calc_pattern = r"<<(.*?)=(.*?)>>"
    calculations = re.findall(calc_pattern, answer_text)

    for expression, result in calculations:
        result = result.strip()
        expression = expression.strip()

        # Parse the expression (assuming format: num1 operator num2)
        # This regex captures numbers and operators
        parts_pattern = r"(\d+)\s*([\+\-\*\/])\s*(\d+)"
        match = re.match(parts_pattern, expression)
        if match:
            num1, operator, num2 = match.groups()

            # Add all numbers as nodes
            graph.add_node(
                num1, is_input=True, entity_name=variable_meanings.get(num1, "")
            )
            graph.add_node(
                num2, is_input=True, entity_name=variable_meanings.get(num2, "")
            )
            graph.add_node(
                result, is_input=False, entity_name=variable_meanings.get(result, "")
            )

            # Add the computation edge
            graph.add_edge(num1, num2, operator, result)

    return graph


def visualize_graph(graph: ComputationGraph) -> None:
    """Print a simple visualization of the computation graph"""
    print("Computation Graph:")
    print("\nNodes:")
    for value, node in graph.nodes.items():
        node_type = "INPUT" if node.is_input else "COMPUTED"
        print(f"  {value} ({node_type})")

    print("\nEdges (Computations):")
    for target, edges in graph.edges.items():
        for src1, src2, op in edges:
            print(f"  {src1} {op} {src2} = {target}")

    print("\nDependency Chain:")
    for target, edges in graph.edges.items():
        parents = graph.get_parents(target)
        if parents:
            print(f"  {target} depends on: {', '.join(parents)}")


def visualize_graph_graphviz(
    graph: ComputationGraph, output_file: str = "computation_graph"
) -> None:
    """
    Visualize the computation graph using graphviz
    Args:
        graph: ComputationGraph object
        output_file: Name of the output file (without extension)
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("Please install graphviz: pip install graphviz")
        return

    dot = Digraph(comment="Computation Graph")
    dot.attr(rankdir="BT")  # Bottom to top layout

    # Add nodes
    for value, node in graph.nodes.items():
        shape = "box" if node.is_input else "ellipse"
        dot.node(value, value, shape=shape)

    # Add edges with operation labels
    for target, edges in graph.edges.items():
        for src1, src2, op in edges:
            # Create an invisible operation node to show the operator
            op_node = f"{src1}_{op}_{src2}"
            dot.node(op_node, op, shape="circle", style="filled", fillcolor="lightgray")

            # Connect source nodes to operation node
            dot.edge(src1, op_node)
            dot.edge(src2, op_node)
            # Connect operation node to target
            dot.edge(op_node, target)

    # Save the graph without viewing
    dot.render(output_file, view=False, format="png")


def extract_variable_meanings(template, variables):
    """
    Extract semantic meanings for variables from the template structure

    For Butcher Sales example:
    - rate: "hourly meat sales rate"
    - hours: "working hours per day"
    - daily: "daily meat sales"
    - weight: "total meat weight"
    - days: "days needed to sell"
    """
    meanings = {}

    # Map variables to their semantic meanings based on template context
    if template.type_name == "Butcher Sales":
        meanings = {
            str(variables["rate"]): "hourly meat sales rate",
            str(variables["hours"]): "working hours per day",
            str(variables["daily"]): "daily meat sales",
            str(variables["weight"]): "total meat weight",
            str(variables["days"]): "days needed to sell",
        }

    return meanings


def print_ascii_tree(graph: ComputationGraph) -> None:
    """
    Print an ASCII representation of the computation graph
    """

    def find_final_result():
        # Find nodes that are not used as inputs in any calculation
        all_sources = set()
        for edges in graph.edges.values():
            for src1, src2, _ in edges:
                all_sources.add(src1)
                all_sources.add(src2)

        final_results = set(graph.nodes.keys()) - all_sources
        return list(final_results)[0] if final_results else None

    def print_node_recursive(node_value: str, depth: int = 0, visited=None):
        if visited is None:
            visited = set()

        if node_value in visited:
            return
        visited.add(node_value)

        # Print current node
        prefix = "    " * depth
        node_type = "(INPUT)" if graph.nodes[node_value].is_input else "(COMPUTED)"
        print(f"{prefix}{node_value} {node_type}")

        # Print operations and child nodes
        for src1, src2, op in graph.edges[node_value]:
            print(f"{prefix}├── {op} operation")
            print(f"{prefix}│   ├── {src1}")
            print(f"{prefix}│   └── {src2}")

    # Start from the final result
    final_result = find_final_result()
    if final_result:
        print("Computation Tree (top-down):")
        print_node_recursive(final_result)


# Example usage
from template_v2 import task_templates
import random

index = random.randint(0, len(task_templates) - 1)
index = 5 ## zhuoy
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
example = template.generate(setting)
print(example)

variables = template.variable_generator(setting)
answer_dict = template.answer_generator(variables)
variables.update(answer_dict)

answer_text = template.deduction_template.format(**variables)

# answer = example["answer"]
# variables = example["variables"]

# answer = """Answer: Bob sells 5kg of meat every hour they works.
# In 9 hours they will sell 5 x 9 = <<5*9=45>>45kg of meat.
# Bob will need to sell 540kg of meat.
# 540kg / 45kg = <<540/45=12>>12 days"""
print(answer_text)
graph = parse_computation_graph(answer_text, template, variables)

print_ascii_tree(graph)
visualize_graph_graphviz(graph)
pds()
