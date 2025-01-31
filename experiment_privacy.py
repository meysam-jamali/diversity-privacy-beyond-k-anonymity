# privacy_experiments.py

import os
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init
from k_anonymity import apply_k_anonymity
from l_diversity import basic_l_diversity, entropy_l_diversity, recursive_l_diversity

# Initialize colorama
init(autoreset=True)

# Global storage for results
experiment_results = []

def run_privacy_experiments(dataset, quasi_identifiers, sensitive_attr, max_levels, hierarchies, k_values, l_values):
    """
    Run privacy experiments, check k-anonymity with ℓ-diversity conditions, and display improvements.
    """
    for k in k_values:
        # Apply k-anonymity
        print(Fore.CYAN + f"\nApplying k-anonymity with k={k}..." + Style.RESET_ALL)
        k_anonymized_data = apply_k_anonymity(dataset, quasi_identifiers, k)

        # Initialize counters for group checks under ℓ-diversity
        k_basic_pass_count, k_entropy_pass_count, k_recursive_pass_count = 0, 0, 0
        k_total_groups = len(k_anonymized_data.groupby(quasi_identifiers))

        for l in l_values:
            print(Fore.CYAN + f"\nRunning ℓ-diversity checks for k={k}, l={l}..." + Style.RESET_ALL)

            # Perform ℓ-diversity checks within the k-anonymized groups
            total_groups = 0
            basic_pass_count = entropy_pass_count = recursive_pass_count = 0

            for _, group in k_anonymized_data.groupby(quasi_identifiers):
                total_groups += 1

                # Basic, Entropy, and Recursive ℓ-Diversity checks
                if basic_l_diversity(group, sensitive_attr, l):
                    basic_pass_count += 1
                if entropy_l_diversity(group, sensitive_attr, l)[0]:
                    entropy_pass_count += 1
                if recursive_l_diversity(group, sensitive_attr, l)[0]:
                    recursive_pass_count += 1

            # Calculate pass rates for each type
            basic_pass_rate = (basic_pass_count / total_groups) * 100
            entropy_pass_rate = (entropy_pass_count / total_groups) * 100
            recursive_pass_rate = (recursive_pass_count / total_groups) * 100

            # Track how many **k-anonymity** groups satisfied each ℓ-diversity type
            k_basic_pass_count += basic_pass_count
            k_entropy_pass_count += entropy_pass_count
            k_recursive_pass_count += recursive_pass_count

            # Calculate improvements from **k-anonymity** to **ℓ-diversity**
            improvement_basic = basic_pass_rate - 100.0  # 100% by default for k-anonymity's own success
            improvement_entropy = entropy_pass_rate - 100.0
            improvement_recursive = recursive_pass_rate - 100.0

            # Save results
            experiment_results.append({
                "k": k,
                "l": l,
                "K-Anonymity": "100.00%",  # Always 100% under its own definition
                "Basic ℓ-Diversity": f"{basic_pass_rate:.2f}%",
                "Entropy ℓ-Diversity": f"{entropy_pass_rate:.2f}%",
                "Recursive ℓ-Diversity": f"{recursive_pass_rate:.2f}%",
                "Improvement (Basic)": color_value(improvement_basic),
                "Improvement (Entropy)": color_value(improvement_entropy),
                "Improvement (Recursive)": color_value(improvement_recursive),
            })

    # Display the final table
    display_privacy_results()

def color_value(value):
    """Apply color formatting to improvement values."""
    if value > 0:
        return Fore.GREEN + f"+{value:.2f}%" + Style.RESET_ALL
    elif value == 0:
        return Fore.YELLOW + f"{value:.2f}%" + Style.RESET_ALL
    else:
        return Fore.RED + f"{value:.2f}%" + Style.RESET_ALL

def display_privacy_results():
    """Display results in a tabular format with color-coded improvements."""
    results_df = pd.DataFrame(experiment_results)

    # Format and print results
    print("\n🔍 Privacy Experiment Results (Summary Table)\n")
    print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))

def main():
    """Main function to execute privacy experiments."""
    # Load dataset
    dataset_path = "./data/original_dataset/adult_dataset.csv"
    if not os.path.exists(dataset_path):
        print(Fore.RED + f"❌ Dataset not found at '{dataset_path}'. Please provide a valid path." + Style.RESET_ALL)
        return

    dataset = pd.read_csv(dataset_path)
    print(Fore.GREEN + "✅ Dataset loaded successfully.\n" + Style.RESET_ALL)

    # Define parameters
    quasi_identifiers = ["Age", "Gender", "Education", "Work Class"]
    sensitive_attr = "Occupation"
    k_values = [2, 4, 6, 8]
    l_values = [2, 4, 6, 8]
    max_levels = [4, 1, 1, 2]
    hierarchies = ["suppression", "taxonomy", "taxonomy", "taxonomy"]

    # Run experiments
    run_privacy_experiments(
        dataset=dataset,
        quasi_identifiers=quasi_identifiers,
        sensitive_attr=sensitive_attr,
        max_levels=max_levels,
        hierarchies=hierarchies,
        k_values=k_values,
        l_values=l_values
    )

    # Save results to CSV
    output_csv = "./data/privacy_experiment_results.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(experiment_results).to_csv(output_csv, index=False)
    print(Fore.GREEN + f"\n✅ Privacy experiments completed! Results saved to '{output_csv}'." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
