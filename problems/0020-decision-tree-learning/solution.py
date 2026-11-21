import math
from collections import Counter
import numpy as np


def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    counts = Counter(labels) 
    total = len(labels)
    entropy = 0 
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    labels = [example[target_attr] for example in examples]
    parent_entropy = calculate_entropy(labels)
    attr_values = [example[attr] for example in examples]
    unique_values = set(attr_values)
    weighted_entropy = 0
    for value in sorted(unique_values):
        subset = [example for example in examples if example[attr] == value]
        weight = len(subset) / len(examples)  
        subset_labels = [example[target_attr] for example in subset]
        weighted_entropy += weight * calculate_entropy(subset_labels)
    return parent_entropy - weighted_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = [example[target_attr] for example in examples]
    counts = Counter(labels)
    # Get all classes that have the max count
    max_count = max(counts.values())
    # Filter to only classes with that count using a list comprehension
    top_classes = [cls for cls, count in counts.items() if count == max_count]
    # Pick the alphabetically first one using min(), which works alphabetically on strings
    return min(top_classes)

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""

    labels = [example[target_attr] for example in examples]
    counts = Counter(labels)
    if len(examples) == 0 or len(attributes) == 0:
     return majority_class(examples, target_attr)
    if len(counts) == 1 :  # This says: "if there's only one unique class, return it"
     return labels[0]  
    best_attr = max(attributes, key=lambda attr: calculate_information_gain(examples, attr, target_attr))  # IG for each one and pick the best one
    remaining_attrs = [attr for attr in attributes if attr != best_attr]
    tree = {best_attr: {}} 
    unique_values = sorted(set(example[best_attr] for example in examples))
    for value in unique_values :
       subset = [example for example in examples if example[best_attr] == value]
       tree[best_attr][value] = learn_decision_tree(subset, remaining_attrs, target_attr)
    return tree
