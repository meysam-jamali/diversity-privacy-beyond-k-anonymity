# Import necessary libraries
import pandas as pd
import numpy as np
import random
import time
import os
from colorama import Fore, Style
import matplotlib.pyplot as plt
from k_anonymity import apply_k_anonymity
import l_diversity 

# Utility functions for metrics
def calculate_generalization_height(original_data, anonymized_data, quasi_identifiers):
    """Calculate generalization height as the reduction in unique values per quasi-identifier."""
    heights = {}
    for qi in quasi_identifiers:
        original_unique = original_data[qi].nunique()
        anonymized_unique = anonymized_data[qi].nunique()
        heights[qi] = original_unique - anonymized_unique
    print(f"Generalization Heights per QI: {heights}")
    return np.mean(list(heights.values()))  # Return the average height

def calculate_average_qblock_size(anonymized_data, quasi_identifiers):
    """Calculate the average size of q*-blocks."""
    try:
        grouped = anonymized_data.groupby(quasi_identifiers)
        avg_size = np.mean([len(group) for _, group in grouped])
        print(f"Average Q-Block Size: {avg_size}")
        return avg_size
    except Exception as e:
        print(f"Error calculating Average Q-Block Size: {e}")
        return np.nan

def calculate_discernibility_metric(anonymized_data, quasi_identifiers):
    """Calculate the discernibility metric."""
    grouped = anonymized_data.groupby(quasi_identifiers)
    return sum([len(group)**2 for _, group in grouped])


def add_combined_columns(data, quasi_identifiers, max_levels, hierarchies, generalization_levels):
    """Ensure that combined quasi-identifier columns are added to the dataset and generalization levels are updated."""
    if "Age" in data.columns and "Gender" in data.columns:
        if "Age + Gender" not in data.columns:
            data["Age + Gender"] = data["Age"].astype(str) + " + " + data["Gender"]
            quasi_identifiers.append("Age + Gender")
            max_levels.append(3)
            hierarchies.append("taxonomy")
            generalization_levels["Age + Gender"] = 3  # Add the new column to generalization levels

    if "Zipcode" in data.columns and "Order Date" in data.columns:
        if "Zipcode + Order Date" not in data.columns:
            data["Zipcode + Order Date"] = data["Zipcode"].astype(str) + " + " + data["Order Date"]
            quasi_identifiers.append("Zipcode + Order Date")
            max_levels.append(3)
            hierarchies.append("taxonomy")
            generalization_levels["Zipcode + Order Date"] = 3  # Add the new column to generalization levels

# Run Utility Experiments
def run_utility_experiments(dataset, dataset_name, quasi_identifiers, sensitive_attr, ks, ls, c_values, output_dir, generalization_levels=None, max_levels=None, hierarchies=None):
    """Run utility experiments for k-anonymity and ℓ-diversity with metrics calculations."""
    
    if max_levels is None:
        raise ValueError("max_levels must be provided to apply ℓ-diversity.")
    if generalization_levels is None:
        generalization_levels = {qi: 1 for qi in quasi_identifiers}

    results = []
    os.makedirs(output_dir, exist_ok=True)

    for k in ks:
        for l in ls:
            for c in c_values:
                print(Fore.CYAN + f"Processing k={k}, l={l}, c={c} for {dataset_name}..." + Style.RESET_ALL)

                try:
                    # Apply k-anonymity
                    k_anonymized = apply_k_anonymity(dataset, quasi_identifiers, k)

                    # Apply ℓ-diversity once
                    l_diverse_data = l_diversity.apply_l_diversity(
                        k_anonymized, quasi_identifiers, sensitive_attr, l, max_levels=max_levels, hierarchies=hierarchies
                    )

                    # Initialize pass counts
                    basic_pass_count = 0
                    entropy_pass_count = 0
                    recursive_pass_count = 0
                    total_groups = 0

                    # Iterate over groups to calculate ℓ-diversity metrics
                    groups = l_diverse_data.groupby(quasi_identifiers)
                    for group_name, group in groups:
                        total_groups += 1

                        # Perform ℓ-diversity checks
                        basic_pass = l_diversity.basic_l_diversity(group, sensitive_attr, l)
                        entropy_pass, entropy_value, log_l = l_diversity.entropy_l_diversity(group, sensitive_attr, l)
                        recursive_pass, top_value, sum_of_others = l_diversity.recursive_l_diversity(group, sensitive_attr, l)

                        # Update pass counts
                        if basic_pass:
                            basic_pass_count += 1
                        if entropy_pass:
                            entropy_pass_count += 1
                        if recursive_pass:
                            recursive_pass_count += 1

                    # **Updated Metrics Calculation**
                    metrics = {
                        "k": k,
                        "l": l,
                        "c": c,
                        "Basic ℓ-Diversity Satisfaction": f"{basic_pass_count / total_groups:.2%}",
                        "Entropy ℓ-Diversity Satisfaction": f"{entropy_pass_count / total_groups:.2%}",
                        "Recursive ℓ-Diversity Satisfaction": f"{recursive_pass_count / total_groups:.2%}",
                        "Generalization Height (k-Anonymity)": calculate_generalization_height(dataset, k_anonymized, quasi_identifiers),
                        "Generalization Height (Entropy)": calculate_generalization_height(dataset, l_diverse_data, quasi_identifiers) if l_diverse_data is not None else 0,
                        "Generalization Height (Recursive)": calculate_generalization_height(dataset, l_diverse_data, quasi_identifiers) if l_diverse_data is not None else 0,
                        "Min. Avg. Group Size (Entropy)": calculate_average_qblock_size(l_diverse_data, quasi_identifiers) if l_diverse_data is not None else 0,
                        "Min. Avg. Group Size (Recursive)": calculate_average_qblock_size(l_diverse_data, quasi_identifiers) if l_diverse_data is not None else 0,
                        "Discernibility Metric (k-Anonymity)": calculate_discernibility_metric(k_anonymized, quasi_identifiers),
                        "Discernibility Metric (Entropy)": calculate_discernibility_metric(l_diverse_data, quasi_identifiers) if l_diverse_data is not None else 0,
                        "Discernibility Metric (Recursive)": calculate_discernibility_metric(l_diverse_data, quasi_identifiers) if l_diverse_data is not None else 0,
                    }

                    print(Fore.YELLOW + f"Metrics for k={k}, l={l}, c={c}:" + Style.RESET_ALL)
                    for key, value in metrics.items():
                        print(f"{key}: {value}")

                    results.append(metrics)

                except Exception as e:
                    print(Fore.RED + f"Error in utility experiment for k={k}, l={l}, c={c}: {e}" + Style.RESET_ALL)

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    print(Fore.GREEN + "Final Utility Results DataFrame:" + Style.RESET_ALL)
    print(results_df.head())
    return results_df



def run_performance_experiments_fixed_k_l(
    dataset, dataset_name, quasi_identifiers, sensitive_attr, ks_ls, qi_levels, generalization_levels, output_dir, max_levels=None, hierarchies=None
):
    """
    Run performance experiments for k = l, where k and l are in ks_ls.
    """
    if max_levels is None or hierarchies is None:
        raise ValueError("Both max_levels and hierarchies must be provided.")

    results = []
    os.makedirs(output_dir, exist_ok=True)

    for k in ks_ls:  # Iterate over provided k and l values
        for qi_level in qi_levels:
            print(Fore.GREEN + f"Running performance experiment for k = l = {k}, QI level = {qi_level}" + Style.RESET_ALL)

            # Subset of quasi-identifiers and levels based on current experiment configuration
            qis_subset = quasi_identifiers[:qi_level]
            max_levels_subset = max_levels[:qi_level]
            hierarchies_subset = hierarchies[:qi_level]

            try:
                # Step 1: Apply k-anonymity
                k_start_time = time.time()
                k_anonymized = apply_k_anonymity(dataset, qis_subset, k)
                k_execution_time = time.time() - k_start_time

                # Step 2: Apply entropy ℓ-diversity on the k-anonymized dataset
                entropy_start_time = time.time()
                entropy_l_diverse = l_diversity.apply_l_diversity(
                    data=k_anonymized,
                    quasi_identifiers=qis_subset,
                    sensitive_attr=sensitive_attr,
                    l=k,
                    max_levels=max_levels_subset,
                    hierarchies=hierarchies_subset,
                    redistributed_records=None,
                    max_iterations=10
                )
                entropy_execution_time = time.time() - entropy_start_time

                # Step 3: Apply recursive ℓ-diversity on the k-anonymized dataset
                recursive_start_time = time.time()
                recursive_l_diverse = l_diversity.apply_l_diversity(
                    data=k_anonymized,
                    quasi_identifiers=qis_subset,
                    sensitive_attr=sensitive_attr,
                    l=k,
                    max_levels=max_levels_subset,
                    hierarchies=hierarchies_subset,
                    redistributed_records=None,
                    max_iterations=10
                )
                recursive_execution_time = time.time() - recursive_start_time

                # Collect and store performance results
                results.append({
                    "k": k,
                    "l": k,
                    "QI Level": qi_level,
                    "Execution Time (k-Anonymity)": k_execution_time,
                    "Execution Time (Entropy ℓ-Diversity)": entropy_execution_time,
                    "Execution Time (Recursive ℓ-Diversity)": recursive_execution_time
                })

            except Exception as e:
                print(Fore.RED + f"Error processing k = l = {k}, QI level = {qi_level}: {e}" + Style.RESET_ALL)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, f"{dataset_name}_performance_results.csv")
    results_df.to_csv(results_file, index=False)
    print(Fore.GREEN + f"Performance results saved to {results_file}" + Style.RESET_ALL)

    return results_df


# Save Datasets
def save_datasets(output_dir, dataset_name, k, l, c, k_anonymized, entropy_l_diverse, recursive_l_diverse, qis_subset):
    qi_label = "_".join(qis_subset)
    os.makedirs(output_dir, exist_ok=True)

    k_file = os.path.join(output_dir, f"{dataset_name}_k{k}_QI{qi_label}_anonymized.csv")
    entropy_file = os.path.join(output_dir, f"{dataset_name}_k{k}_l{l}_QI{qi_label}_entropy.csv")
    recursive_file = os.path.join(output_dir, f"{dataset_name}_k{k}_l{l}_c{c}_QI{qi_label}_recursive.csv")

    k_anonymized.to_csv(k_file, index=False)
    entropy_l_diverse.to_csv(entropy_file, index=False)
    recursive_l_diverse.to_csv(recursive_file, index=False)

    print(Fore.GREEN + f"Saved datasets for k={k}, l={l}, c={c}..." + Style.RESET_ALL)

def visualize_utility_results(results, output_dir, dataset_name):
    """Visualize utility results (no execution time metrics)."""
    os.makedirs(output_dir, exist_ok=True)

    # Validate required columns
    required_columns = [
        "Generalization Height (k-Anonymity)",
        "Generalization Height (Entropy)",
        "Generalization Height (Recursive)",
        "Min. Avg. Group Size (Entropy)",
        "Min. Avg. Group Size (Recursive)",
        "Discernibility Metric (k-Anonymity)",
        "Discernibility Metric (Entropy)",
        "Discernibility Metric (Recursive)"
    ]
    for col in required_columns:
        if col not in results.columns:
            raise ValueError(f"Column '{col}' is missing from the results DataFrame.")

    # Replace NaN or None with 0 for visualization
    results = results.fillna(0)

    # Define bar positions and width
    x_values = [2, 4, 6, 8]  # Only k = l values for utility visualization
    bar_width = 0.25
    bar_positions = np.arange(len(x_values))

    # 1. Generalization Height Plot
    plt.figure(figsize=(8, 5))
    plt.bar(bar_positions - bar_width, results["Generalization Height (k-Anonymity)"][:4], width=bar_width, label="k-Anonymity", color="black")
    plt.bar(bar_positions, results["Generalization Height (Entropy)"][:4], width=bar_width, label="Entropy ℓ-diversity", color="gray")
    plt.bar(bar_positions + bar_width, results["Generalization Height (Recursive)"][:4], width=bar_width, label="Recursive ℓ-diversity", color="white", edgecolor="black", hatch="/")
    plt.title(f"{dataset_name}: Generalization Height")
    plt.xlabel("k = l")
    plt.ylabel("Generalization Height")
    plt.xticks(bar_positions, x_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_generalization_height.png"))
    plt.show()

    # 2. Minimum Average Group Size Plot
    plt.figure(figsize=(8, 5))
    plt.bar(bar_positions - bar_width / 2, results["Min. Avg. Group Size (Entropy)"][:4], width=bar_width, label="Entropy ℓ-diversity", color="gray")
    plt.bar(bar_positions + bar_width / 2, results["Min. Avg. Group Size (Recursive)"][:4], width=bar_width, label="Recursive ℓ-diversity", color="white", edgecolor="black", hatch="/")
    plt.title(f"{dataset_name}: Minimum Average Group Size")
    plt.xlabel("k = l")
    plt.ylabel("Minimum Average Group Size")
    plt.xticks(bar_positions, x_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_min_avg_group_size.png"))
    plt.show()

    # 3. Discernibility Metric Plot
    plt.figure(figsize=(8, 5))
    plt.bar(bar_positions - bar_width, results["Discernibility Metric (k-Anonymity)"][:4], width=bar_width, label="k-Anonymity", color="black")
    plt.bar(bar_positions, results["Discernibility Metric (Entropy)"][:4], width=bar_width, label="Entropy ℓ-diversity", color="gray")
    plt.bar(bar_positions + bar_width, results["Discernibility Metric (Recursive)"][:4], width=bar_width, label="Recursive ℓ-diversity", color="white", edgecolor="black", hatch="/")
    plt.title(f"{dataset_name}: Discernibility Metric")
    plt.xlabel("k = l")
    plt.ylabel("Discernibility Metric")
    plt.yscale("log")
    plt.xticks(bar_positions, x_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_discernibility_metric.png"))
    plt.show()

    print(f"Utility visualizations saved for {dataset_name}.")

# Visualize performance results
def visualize_performance_results(results, output_dir, dataset_name):
    """
    Visualize performance results for k-anonymity and ℓ-diversity.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the x-axis values and filter results
    x_values = [2, 4, 6, 8]  # Fixed k = l values
    filtered_results = results[(results["k"] == results["l"]) & (results["k"].isin(x_values))]

    # Aggregate data to ensure unique x-values (average execution times for same k=l)
    aggregated_results = (
        filtered_results.groupby("k")
        .agg({
            "Execution Time (k-Anonymity)": "mean",
            "Execution Time (Entropy ℓ-Diversity)": "mean",
            "Execution Time (Recursive ℓ-Diversity)": "mean",
        })
        .reset_index()
    )

    # Ensure we have valid data
    if aggregated_results.empty:
        print(f"No valid data to visualize for {dataset_name} performance results.")
        return

    # 1. Plot Execution Time (k-Anonymity)
    plt.figure(figsize=(8, 5))
    plt.plot(
        aggregated_results["k"], aggregated_results["Execution Time (k-Anonymity)"],
        label="k-Anonymity", marker="o", color="blue"
    )
    plt.title(f"{dataset_name}: Execution Time (k-Anonymity)")
    plt.xlabel("k = l")
    plt.ylabel("Execution Time (Seconds)")
    plt.xticks(x_values)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_execution_time_k_anonymity.png"))
    plt.show()

    # 2. Plot Execution Time (Entropy ℓ-Diversity)
    plt.figure(figsize=(8, 5))
    plt.plot(
        aggregated_results["k"], aggregated_results["Execution Time (Entropy ℓ-Diversity)"],
        label="Entropy ℓ-Diversity", marker="o", color="orange"
    )
    plt.title(f"{dataset_name}: Execution Time (Entropy ℓ-Diversity)")
    plt.xlabel("k = l")
    plt.ylabel("Execution Time (Seconds)")
    plt.xticks(x_values)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_execution_time_entropy.png"))
    plt.show()

    # 3. Plot Execution Time (Recursive ℓ-Diversity)
    plt.figure(figsize=(8, 5))
    plt.plot(
        aggregated_results["k"], aggregated_results["Execution Time (Recursive ℓ-Diversity)"],
        label="Recursive ℓ-Diversity", marker="o", color="green"
    )
    plt.title(f"{dataset_name}: Execution Time (Recursive ℓ-Diversity)")
    plt.xlabel("k = l")
    plt.ylabel("Execution Time (Seconds)")
    plt.xticks(x_values)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_execution_time_recursive.png"))
    plt.show()

    print(f"Performance visualizations saved for {dataset_name}.")

def main():
    """Run experiments for Adult and Lands End datasets."""
    try:
        # Load datasets
        adult_df = pd.read_csv("./data/original_dataset/adult_dataset.csv")
        lands_end_df = pd.read_csv("./data/original_dataset/lands_end_dataset.csv")
        print(Fore.GREEN + "Datasets loaded successfully." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error loading datasets: {e}" + Style.RESET_ALL)
        return

    # Ensure Age column in Adult dataset is numeric
    if "Age" in adult_df.columns:
        adult_df["Age"] = pd.to_numeric(adult_df["Age"], errors="coerce").fillna(-1)

    # Define parameters
    ks = [2, 4, 6, 8]  # Values of k for utility and performance experiments
    ls = [2, 4, 6, 8]  # Values of l for ℓ-diversity
    c_values = [3]  # Fixed c for recursive ℓ-diversity
    output_dir = "./data/results"

    # Quasi-identifiers and sensitive attributes
    adult_qis = ["Age", "Gender", "Education", "Work Class"]
    lands_end_qis = ["Zipcode", "Gender", "Style", "Price"]
    adult_sensitive_attr = "Occupation"
    lands_end_sensitive_attr = "Cost"

    # Generalization hierarchies and maximum levels (without QI range levels)
    adult_hierarchies = ["suppression", "taxonomy", "taxonomy", "taxonomy"]
    adult_max_levels = [4, 1, 1, 2]
    lands_end_hierarchies = ["suppression", "taxonomy", "taxonomy", "taxonomy"]
    lands_end_max_levels = [2, 2, 2, 2]

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # ---- Adult Dataset ----
    print(Fore.BLUE + "Running experiments for Adult Dataset..." + Style.RESET_ALL)
    
    # Utility experiments
    adult_utility_results = run_utility_experiments(
        dataset=adult_df,
        dataset_name="Adult",
        quasi_identifiers=adult_qis,
        sensitive_attr=adult_sensitive_attr,
        ks=ks,
        ls=ls,
        c_values=c_values,
        output_dir=output_dir,
        max_levels=adult_max_levels,
        hierarchies=adult_hierarchies,
    )
    # Performance experiments
    # Run Performance Experiments
    adult_performance_results = run_performance_experiments_fixed_k_l(
        dataset=adult_df,
        dataset_name="Adult",
        quasi_identifiers=adult_qis,
        sensitive_attr=adult_sensitive_attr,
        ks_ls=ks,
        qi_levels=range(1, len(adult_qis) + 1),  # Define levels from 1 to total number of QIs
        generalization_levels={qi: 1 for qi in adult_qis},  # Initialize generalization levels
        output_dir=output_dir,
        max_levels=adult_max_levels,
        hierarchies=adult_hierarchies,
    )

    # Save and visualize results
    adult_utility_results.to_csv(os.path.join(output_dir, "adult_utility_results.csv"), index=False)
    adult_performance_results.to_csv(os.path.join(output_dir, "adult_performance_results.csv"), index=False)
    visualize_utility_results(adult_utility_results, output_dir, "Adult")
    visualize_performance_results(adult_performance_results, output_dir, "Adult")

    # # ---- Lands End Dataset ----
    # print(Fore.BLUE + "Running experiments for Lands End Dataset..." + Style.RESET_ALL)

    # # Utility experiments
    # lands_end_utility_results = run_utility_experiments(
    #     dataset=lands_end_df,
    #     dataset_name="Lands End",
    #     quasi_identifiers=lands_end_qis,
    #     sensitive_attr=lands_end_sensitive_attr,
    #     ks=ks,
    #     ls=ls,
    #     c_values=c_values,
    #     output_dir=output_dir,
    #     max_levels=lands_end_max_levels,
    #     hierarchies=lands_end_hierarchies,
    # )
    # # Performance experiments
    # lands_end_performance_results = run_performance_experiments_fixed_k_l(
    #     dataset=lands_end_df,
    #     dataset_name="Lands End",
    #     quasi_identifiers=lands_end_qis,
    #     sensitive_attr=lands_end_sensitive_attr,
    #     ks_ls=ks,
    #     output_dir=output_dir,
    #     max_levels=lands_end_max_levels,
    #     hierarchies=lands_end_hierarchies,
    # )
    # # Save and visualize results
    # lands_end_utility_results.to_csv(os.path.join(output_dir, "lands_end_utility_results.csv"), index=False)
    # lands_end_performance_results.to_csv(os.path.join(output_dir, "lands_end_performance_results.csv"), index=False)
    # visualize_utility_results(lands_end_utility_results, output_dir, "Lands End")
    # visualize_performance_results(lands_end_performance_results, output_dir, "Lands End")

main()
