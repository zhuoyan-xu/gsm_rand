import os
from typing import Optional, List, Dict, Union, Tuple
import dataclasses
import sys
import random
import numpy as np
import warnings

"""
One way to generate random arithmetic expression is given by the link:
https://stackoverflow.com/questions/6881170/is-there-a-way-to-autogenerate-valid-arithmetic-expressions
However, I sorta don't like it because of the use of recursive definition and uncontrolled length and parentheses
"""

def gen_prob(prob: Union[list, None], L: int):
    """
    simple function that returns a uniform probability vector in np.array
    """
    if prob is None:
        prob = [1/L] * L
    else:
        assert len(prob) == L, "length does not match."
    return np.array(prob) / np.sum(np.array(prob))
    
def generate_random_dyck_1(n: int):
    """
    Generate a random Dyck-1 string with n pairs of parentheses.
    Args:
        n (int): The number of pairs of parentheses.
    Returns:
        encoding: A list of binary numbers encoding open/close parentheses, length 2*n
        string: A random Dyck-1 string.
    """
    # Initialize counts for open and close parentheses
    open_count = 0
    close_count = 0
    encoding = []
    string = []

    # Generate the string step by step
    for _ in range(2 * n):  # Total length will be 2 * n
        if open_count < n and (close_count == open_count or random.choice([True, False])):
            # Add '(' if not all open parentheses are used
            string.append('(') 
            encoding.append(0)
            open_count += 1
        else:
            # Add ')' if it balances an open parenthesis
            string.append(')')
            encoding.append(1)
            close_count += 1

    return encoding, ''.join(string)

def gen_default_tokernizer():
    vocab = {
        "variables": list(range(10)), # each of 10 digits is treated as a token
        "operators": [10, 11], # representing "+", "-"
        "parentheses": [12, 13],  # representing "(", ")"
        "deduction_symbol": [14], 
        "padding": [15]
    }
    vocab_str = {
        "variables": [str(i) for i in range(10)], # each of 10 digits is treated as a token
        "operators": ["+", "-"], # representing "+", "-"
        "parentheses": ["(", ")"],  # representing "(", ")" 
        "deduction_symbol": ["  >>>\n"], 
        "padding": ["P"]
    }
    map = vocab_str["variables"] + vocab_str["operators"] + vocab_str["parentheses"] + vocab_str["deduction_symbol"] + vocab_str["padding"]
    return {"vocab": vocab, "vocab_str": vocab_str, "map": map}

class simpleReasoning():
    """
    Simple arithmetic expression involving modular addition/subtraction (mod 10)
    """
    def __init__(
            self, 
            tokenizer: Dict,
            max_variables: Optional[int] = 8, # length of the formula (number of variables)
            max_parenthesis: Optional[int] = 4,
            num_variables_samp_prob: Optional[list] = None,
            num_parentheses_samp_prob: Optional[list] = None,
            variable_samp_prob: Optional[list] = None,
            operator_samp_prob: Optional[list] = None,
            max_seq_len: Optional[int] = 64,
            random_seed: Optional[int] = 42
            ):
        self.vocab = tokenizer["vocab"] # self.vocab is a dictionary storing the indices for each token in the expression
        self.map = tokenizer["map"]
        self.max_variables = max_variables
        self.max_parenthesis = max_parenthesis
        self.num_variables_samp_prob = num_variables_samp_prob
        self.num_parentheses_samp_prob = num_parentheses_samp_prob
        self.variable_samp_prob = variable_samp_prob
        self.operator_samp_prob = operator_samp_prob
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.mod = len(self.vocab["variables"])
        self.sample_size = None

    def map_ids_to_str(self, ids: np.ndarray) -> str:
        return "".join([self.map[j] for j in ids])

    def sample(self, num_steps: Optional[int] = 3):
        """
        Generate num_steps number of random arithmetic expressions, 
        where each expression is a list of token indices of variables, operators, and parentheses
        An example of one formula (in string): "((0-(6-9-(0-4-7)))-7+8)"
        """
        self.sample_size = num_steps
        V1 = self.vocab["variables"] # all variables
        V2 = self.vocab["operators"] # all operators
        V3 = self.vocab["parentheses"] # all operators
        p_var = gen_prob(self.variable_samp_prob, len(V1))
        p_op = gen_prob(self.operator_samp_prob, len(V2))
        p_var_len = gen_prob(self.num_variables_samp_prob, self.max_variables-1) # prob vector on [2,3...max_variables]
        p_paren_len = gen_prob(self.num_parentheses_samp_prob, self.max_parenthesis)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        result = []
        while len(result) < num_steps:
            # generate variables and operators
            L1 = np.random.choice(np.arange(2,self.max_variables+1), p=p_var_len) # number of variables
            variables = np.random.choice(V1, size=L1, replace=True, p=p_var)
            operators = np.random.choice(V2, size=L1-1, replace=True, p=p_op)
            # next, generate parentheses
            L2 = L1 + 1 # number of parenthesis pairs
            while L2 >= L1:
                L2 = np.random.choice(np.arange(1,self.max_parenthesis+1), p=p_paren_len)
            paren_encoding, _ = generate_random_dyck_1(L2)
            
            # then, randomly assign an index for each parenthesis, where index is in [0, L1]
            K = np.sum([(paren_encoding[k]==0 and paren_encoding[k+1]==1) for k in range(2*L2-1)]) # number of adjacent parentheses
            if L1+1-2*K < 1: # invalid case: too many pairs of adjacent parentheses
                continue
            indices = np.sort(np.random.choice(L1+1-2*K, size=2*L2, replace=True))
            for k in reversed(range(0, 2*L2-1)):
                if paren_encoding[k]==0 and paren_encoding[k+1]==1: 
                    # if () is adjacent in dyck language, need to ensure at least two variables are inserted in between parentheses
                    indices[(k+1):] += 2

            assert len(indices) == 2*L2, "check length of indices"
            assert max(indices) <= L1, "check range of indices"
            # finally, insert parentheses into formula at locations specified by indices
            # start from the plain formula, e.g., 4 + 2 - 9 + 1
            ids = np.array([variables[0]]) # building ids, the list of token indices
            for j in range(L1-1):
                ids = np.concatenate((ids, np.array([operators[j], variables[j+1]])))
            # now insert parentheses
            for k in reversed(range(0, 2*L2)):
                is_open = (paren_encoding[k]==0) # if (, then add to position after the operator
                pos_insert = 2*indices[k]-1
                ids = np.insert(ids, pos_insert+is_open, V3[paren_encoding[k]])
            result.append([variables, operators, ids])
        return result
    
    def eval_one_step(self, ids: np.ndarray) -> int:
        """
        Evaluate a simple arithmetic modular expression
        """
        try:
            result = eval(self.map_ids_to_str(ids))
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
        val = self.vocab["variables"][result % self.mod]
        return val
    
    def simplfy(self, ids: np.ndarray) -> np.ndarray:
        """
        One-step simplication of an arithmetic expression.
        The earliest inner parenthesis is evaluated and replaced by the resulting value
        An example: if ids is token indices for "((0-(6-9-(0-4-7)))-7+8)", then 
            output is a shorted token indices "((0-(6-9-9))-7+8)"
        """
        assert len(ids) > 2, "expression too short!"
        indices_open_paren = []
        indices_close_paren = []
        for j, id in enumerate(ids):
            indices_open_paren.append(j) if id == self.vocab["parentheses"][0] else None
            indices_close_paren.append(j) if id == self.vocab["parentheses"][1] else None
        if len(indices_close_paren) == 0: # in the no parenthesis case, simply first operator
            val = self.eval_one_step(ids[0:3])
            return np.concatenate((np.array([val]), ids[3:]))
        id2 = min(indices_close_paren) 
        id1 = max([x for x in indices_open_paren if x < id2])
        if id2 - id1 == 2 or id2 - id1 == 4: # if the parenthesis pair contains
                # only zero/one operator such as "( 4 )" or "( 4 + 5 )", evalute directly
            val = self.eval_one_step(ids[id1:(id2+1)])
        else: # if longer expression such as "( 4 + 5 - 6)", then only evaluate "4 + 5"
            id1, id2 = id1+1, id1+3
            val = self.eval_one_step(ids[id1:(id2+1)])
        return np.concatenate((ids[:id1], np.array([val]), ids[(id2+1):]))
    
    def formatted_sample(self, num_steps: Optional[int] = 3, left_padding = True, discard_warning=False):
        """
        Sample and then format num_steps number of prompts. Each prompt is a formual followed by a one-step simplification. 
        For example: in string, one prompt looks like
                                "(9+((7+5))-1)  >>>
                                (9+(2)-1)"
        In addition, if left_padding is True, then padding tokens are added to the left to reach self.max_seq_len
        """
        result = self.sample(num_steps)
        result_formatted = []
        info = {
            "lens": None,
            "solution_start_ind": None,
            "first_non_padding_idx": None
        }
        deduct = self.vocab["deduction_symbol"][0] # deduction token
        # concantenate formula with its one-step simplification
        for i in range(num_steps):
            ids_formatted = np.concatenate((result[i][2], [deduct], self.simplfy(result[i][2])))
            result_formatted.append(ids_formatted)
        lens = [len(ids) for ids in result_formatted]
        # discard prompts whose lengths exceed the max_seq_len
        if  max(lens) > self.max_seq_len:
            if discard_warning:
                warnings.warn("Some input sequence length exceeeds max_seq_len; discarded those input samples.")
            result_formatted = [ids for ids in result_formatted if len(ids)<=self.max_seq_len]
            self.sample_size = len(result_formatted)
        # add padding tokens to the left
        if left_padding:
            result_formatted = [np.concatenate((np.array(self.vocab["padding"] * (self.max_seq_len-len(ids))), ids)) for ids in result_formatted]
            info["first_non_padding_idx"] = [self.max_seq_len-len(ids) for ids in result_formatted]
        info["lens"] = [len(ids) for ids in result_formatted]
        info["solution_start_ind"] = [np.where(ids == deduct)[0][0]+1 for ids in result_formatted]
        return result_formatted, info
    
    def get_task_details(self) -> Dict:
        return {
            "name": "simple_reasoning",
            "description": "simple arithmetic deduction steps",
            "max_variables": self.max_variables,
            "max_parenthesis": self.max_parenthesis,
            "num_variables_samp_prob": self.num_variables_samp_prob,
            "num_parentheses_samp_prob": self.num_parentheses_samp_prob,
            "variable_samp_prob": self.variable_samp_prob,
            "operator_samp_prob": self.operator_samp_prob,
            "sample_size": self.sample_size
        }
