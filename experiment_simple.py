# Import necessary libraries
import pandas as pd
import numpy as np
import time
import os
from colorama import Fore, Style
from k_anonymity import apply_k_anonymity
from l_diversity_using_k_anonymity_dataset import apply_l_diversity
import matplotlib.pyplot as plt
import numpy as np
import os

# Utility Functions
def calculate_generalization_height(original_data, anonymized_data, quasi_identifiers):
    """Compute average reduction in unique values for quasi-identifiers."""
    heights = {
        qi: original_data[qi].nunique() - anonymized_data[qi].nunique()
        for qi in quasi_identifiers
    }
    return np.mean(list(heights.values()))

def calculate_discernibility_metric(anonymized_data, quasi_identifiers):
    """Calculate the discernibility metric."""
    grouped = anonymized_data.groupby(quasi_identifiers)
    return sum(len(group)**2 for _, group in grouped)

def test_utility_and_performance(dataset, quasi_identifiers, sensitive_attr, k, l, c, max_levels, hierarchies, dataset_name, output_dir):
    """
    Test utility and performance for fixed k, l, and c values.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(Fore.CYAN + f"Testing {dataset_name} with k={k}, l={l}, c={c}..." + Style.RESET_ALL)

    # Step 1: Apply k-Anonymity
    start_time = time.time()
    k_anonymized = apply_k_anonymity(dataset.copy(), quasi_identifiers, k)  # Ensure no in-place modification
    k_time = time.time() - start_time

    # Validate k-Anonymity
    if k_anonymized is None or k_anonymized.empty:
        print(Fore.RED + f"Error: k-Anonymity failed for {dataset_name}. Exiting..." + Style.RESET_ALL)
        return None

    # Step 2: Apply ℓ-Diversity (Entropy)
    start_time = time.time()
    entropy_k_anonymized = k_anonymized.copy()  # Clone the k-anonymized dataset
    entropy_l_diverse = apply_l_diversity(
        entropy_k_anonymized,
        quasi_identifiers,
        sensitive_attr,
        l,
        max_levels=max_levels,
        method="entropy",
        hierarchies=hierarchies,
    )
    entropy_time = time.time() - start_time

    if entropy_l_diverse is None:
        print(Fore.YELLOW + f"ℓ-Diversity (Entropy) could not be applied for {dataset_name}." + Style.RESET_ALL)

    # Step 3: Apply ℓ-Diversity (Recursive)
    start_time = time.time()
    recursive_k_anonymized = k_anonymized.copy()  # Clone the k-anonymized dataset
    recursive_l_diverse = apply_l_diversity(
        recursive_k_anonymized,
        quasi_identifiers,
        sensitive_attr,
        l,
        max_levels=max_levels,
        method="recursive",
        c=c,
        hierarchies=hierarchies,
    )
    recursive_time = time.time() - start_time

    if recursive_l_diverse is None:
        print(Fore.YELLOW + f"ℓ-Diversity (Recursive) could not be applied for {dataset_name}." + Style.RESET_ALL)

    # Step 4: Calculate Utility Metrics
    utility_metrics = {
        "Dataset": dataset_name,
        "k": k,
        "l": l,
        "c": c,
        "k-Anonymity Generalization Height": calculate_generalization_height(dataset, k_anonymized, quasi_identifiers),
        "Entropy ℓ-Diversity Generalization Height": calculate_generalization_height(dataset, entropy_l_diverse, quasi_identifiers) if entropy_l_diverse is not None else None,
        "Recursive ℓ-Diversity Generalization Height": calculate_generalization_height(dataset, recursive_l_diverse, quasi_identifiers) if recursive_l_diverse is not None else None,
        "k-Anonymity Discernibility Metric": calculate_discernibility_metric(k_anonymized, quasi_identifiers),
        "Entropy ℓ-Diversity Discernibility Metric": calculate_discernibility_metric(entropy_l_diverse, quasi_identifiers) if entropy_l_diverse is not None else None,
        "Recursive ℓ-Diversity Discernibility Metric": calculate_discernibility_metric(recursive_l_diverse, quasi_identifiers) if recursive_l_diverse is not None else None,
        "Execution Time (k-Anonymity)": k_time,
        "Execution Time (Entropy)": entropy_time if entropy_l_diverse is not None else None,
        "Execution Time (Recursive)": recursive_time if recursive_l_diverse is not None else None,
    }

    # Step 5: Print Utility Metrics
    for metric, value in utility_metrics.items():
        if value is None:
            print(f"{Fore.YELLOW}{metric}:{Style.RESET_ALL} {Fore.RED}Not Computed{Style.RESET_ALL}")
        elif "Generalization Height" in metric:
            print(f"{Fore.BLUE}{metric}:{Style.RESET_ALL} {Fore.CYAN}{value}{Style.RESET_ALL}")
        elif "Discernibility Metric" in metric:
            print(f"{Fore.GREEN}{metric}:{Style.RESET_ALL} {Fore.MAGENTA}{value}{Style.RESET_ALL}")
        elif "Execution Time" in metric:
            print(f"{Fore.YELLOW}{metric}:{Style.RESET_ALL} {Fore.RED}{value:.4f} seconds{Style.RESET_ALL}")
        else:
            print(f"{Fore.WHITE}{metric}:{Style.RESET_ALL} {value}")

    return utility_metrics


# Main Execution
def main():
    # Load datasets
    try:
        adult_df = pd.read_csv("./data/original_dataset/adult_dataset.csv")
        lands_end_df = pd.read_csv("./data/original_dataset/lands_end_dataset.csv")
        print(Fore.GREEN + "Datasets loaded successfully." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error loading datasets: {e}" + Style.RESET_ALL)
        return

    # Ensure Age column in Adult dataset is numeric
    adult_df["Age"] = pd.to_numeric(adult_df["Age"], errors="coerce")
    adult_df["Age"].fillna(-1, inplace=True)

    # Parameters
    k = 3  # Fixed k
    l = 3  # Fixed l
    c = 2  # Fixed c
    output_dir = "./data/results"

    # Adult Dataset Parameters
    adult_qis = ["Age", "Gender", "Education", "Work Class"]
    adult_sensitive_attr = "Salary Class"
    adult_hierarchies = ["range", "suppression", "taxonomy", "taxonomy"]
    adult_max_levels = [4, 2, 3, 2]

    # Lands End Dataset Parameters
    lands_end_qis = ["Zipcode", "Gender", "Style", "Price"]
    lands_end_sensitive_attr = "Cost"
    lands_end_hierarchies = ["range", "suppression", "taxonomy", "taxonomy"]
    lands_end_max_levels = [3, 2, 2, 2]

    # Test Adult Dataset
    print(Fore.BLUE + "Testing Adult Dataset..." + Style.RESET_ALL)
    adult_results = None
    try:
        adult_results = test_utility_and_performance(
            dataset=adult_df,
            quasi_identifiers=adult_qis,
            sensitive_attr=adult_sensitive_attr,
            k=k,
            l=l,
            c=c,
            max_levels=adult_max_levels,
            hierarchies=adult_hierarchies,
            dataset_name="Adult Dataset",
            output_dir=output_dir,
        )
    except Exception as e:
        print(Fore.RED + f"Error testing Adult Dataset: {e}" + Style.RESET_ALL)

    # Test Lands End Dataset
    print(Fore.BLUE + "Testing Lands End Dataset..." + Style.RESET_ALL)
    lands_end_results = None
    try:
        lands_end_results = test_utility_and_performance(
            dataset=lands_end_df,
            quasi_identifiers=lands_end_qis,
            sensitive_attr=lands_end_sensitive_attr,
            k=k,
            l=l,
            c=c,
            max_levels=lands_end_max_levels,
            hierarchies=lands_end_hierarchies,
            dataset_name="Lands End Dataset",
            output_dir=output_dir,
        )
    except Exception as e:
        print(Fore.RED + f"Error testing Lands End Dataset: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()