import pandas as pd
import numpy as np
from itertools import product
from k_anonymity import apply_k_anonymity
import os
from colorama import Fore, Style, init
from itertools import product
import math
init(autoreset=True)


def generalize_value(value, level, hierarchy, data_min=None, data_max=None):
    """
    Generalize values dynamically based on the hierarchy type.

    Args:
        value: The value to generalize.
        level: The level of generalization (higher levels = more generalization).
        hierarchy: The type of hierarchy (e.g., "range", "round", "taxonomy", "suppression").
        data_min: Minimum value in the column (used for range generalization).
        data_max: Maximum value in the column (used for range generalization).

    Returns:
        The generalized value or "Any" if fully generalized.
    """
    try:
        if pd.isna(value) or value in ["Any", "Suppressed"]:
            return value  # Skip suppressed values

        # Handle range-based generalization for numeric or range-like strings
        if hierarchy == "range":
            # Handle numeric values
            if isinstance(value, (int, float, np.integer, np.floating)):
                if data_min is None or data_max is None:
                    raise ValueError("Range hierarchy requires data_min and data_max.")
                step = (data_max - data_min) / (10 ** level)
                lower_bound = (value // step) * step
                upper_bound = lower_bound + step
                return f"{int(lower_bound)}-{int(upper_bound)}"

            # Handle range-like string values (e.g., "20-29", "30-39")
            elif isinstance(value, str) and "-" in value:
                range_parts = value.split("-")
                if len(range_parts) == 2 and all(part.isdigit() for part in range_parts):
                    lower_bound = int(range_parts[0])
                    upper_bound = int(range_parts[1])
                    # Combine adjacent ranges based on the level
                    step = (upper_bound - lower_bound + 1) * level
                    new_lower = (lower_bound // step) * step
                    new_upper = new_lower + step - 1
                    return f"{new_lower}-{new_upper}"
                else:
                    raise ValueError(f"Non-numeric or invalid range value '{value}' cannot be generalized.")

            else:
                raise ValueError(f"Non-numeric value '{value}' cannot be generalized with 'range'.")

        # Rounding-based generalization for numeric values
        elif hierarchy == "round":
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"Non-numeric value '{value}' cannot be generalized with 'round' hierarchy.")
            step = 10 ** level
            lower_bound = (value // step) * step
            upper_bound = lower_bound + step - 1
            return f"{int(lower_bound)}-{int(upper_bound)}"

        # Taxonomy-based generalization
        elif hierarchy == "taxonomy":
            return f"Level-{level}" if level > 0 else value

        # Suppression-based generalization
        elif hierarchy == "suppression":
            return "Any" if level > 0 else value

        # Default case
        return value

    except Exception as e:
        print(f"{Fore.BLUE}Error generalizing value '{value}': {e}{Style.RESET_ALL}")
        return value


def apply_generalization(data, quasi_identifiers, levels, hierarchies):
    """
    Apply generalization based on levels and hierarchies while preserving original column names.

    Args:
        data (pd.DataFrame): The dataset to be generalized.
        quasi_identifiers (list): List of quasi-identifiers.
        levels (list): Generalization levels for each quasi-identifier.
        hierarchies (list): Generalization hierarchies for each quasi-identifier.

    Returns:
        pd.DataFrame: Generalized dataset with original column names preserved.
    """
    generalized_data = data.copy()

    for i, qi in enumerate(quasi_identifiers):
        level = levels[i]
        hierarchy = hierarchies[i]
        data_min = data[qi].min() if hierarchy == "range" else None
        data_max = data[qi].max() if hierarchy == "range" else None

        def generalize_or_skip(value):
            # Skip suppressed or invalid values
            if value in ["*", "Any"] or pd.isna(value):
                return value
            return generalize_value(value, level, hierarchy, data_min, data_max)

        print(Fore.CYAN + f"Generalizing column '{qi}' with hierarchy '{hierarchy}' at level {level}..." + Style.RESET_ALL)
        generalized_data[qi] = generalized_data[qi].apply(generalize_or_skip)

    # ✅ Ensure column names remain the same
    generalized_data.columns = data.columns

    return generalized_data

# Check k-Anonymity
def check_k_anonymity(data, quasi_identifiers, k):
    """
    Validate if a dataset satisfies k-anonymity and display detailed group size checks.

    Args:
        data (pd.DataFrame): The generalized dataset.
        quasi_identifiers (list): List of quasi-identifiers.
        k (int): The k-anonymity threshold.

    Returns:
        bool: True if k-anonymity is satisfied; False otherwise.
    """
    grouped = data.groupby(quasi_identifiers)
    all_satisfy = True

    print(Fore.YELLOW + "\nChecking k-Anonymity:" + Style.RESET_ALL)
    for group_name, group in grouped:
        group_size = len(group)
        print(Fore.CYAN + f"\nGroup {group_name}: Size = {group_size}, k = {k}" + Style.RESET_ALL)
        
        if group_size < k:
            print(Fore.RED + f"❌ Group {group_name} fails k-anonymity (size {group_size} < k={k})" + Style.RESET_ALL)
            all_satisfy = False
        else:
            print(Fore.GREEN + f"✅ Group {group_name} satisfies k-anonymity (size {group_size} ≥ k={k})" + Style.RESET_ALL)

    return all_satisfy


def check_l_diversity(data, quasi_identifiers, sensitive_attr, l):
    """
    Check if the dataset satisfies ℓ-diversity for all groups.

    Args:
        data (pd.DataFrame): The dataset.
        quasi_identifiers (list): List of quasi-identifier columns.
        sensitive_attr (str): Sensitive attribute column name.
        l (int): ℓ-diversity threshold.

    Returns:
        bool: True if all groups satisfy ℓ-diversity, otherwise False.
    """
    # ✅ Verify that the sensitive attribute exists in the dataset
    if sensitive_attr not in data.columns:
        print(Fore.RED + f"❌ Error: Sensitive attribute '{sensitive_attr}' not found. Available columns: {list(data.columns)}" + Style.RESET_ALL)
        return False

    groups = data.groupby(quasi_identifiers)
    all_satisfy = True

    for group_name, group in groups:
        sensitive_counts = group[sensitive_attr].value_counts()

        # ✅ Ensure at least ℓ distinct values
        num_unique_sensitive_values = len(sensitive_counts)
        if num_unique_sensitive_values >= l:
            print(Fore.GREEN + f"✅ Group {group_name} satisfies ℓ-diversity: {num_unique_sensitive_values} unique sensitive values ≥ ℓ={l}" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"❌ Group {group_name} fails ℓ-diversity: {num_unique_sensitive_values} unique sensitive values < ℓ={l}" + Style.RESET_ALL)
            all_satisfy = False

    return all_satisfy

def generate_lattice(quasi_identifiers, max_levels):
    """
    Generate the generalization lattice for the quasi-identifiers.
    """
    return list(product(*[range(1, max_level + 1) for max_level in max_levels]))

def lattice_search(data, quasi_identifiers, sensitive_attr, l, max_levels, hierarchies, strategy="bottom-up"):
    """
    Perform a lattice search to find the optimal generalization that satisfies ℓ-diversity.

    Args:
        data (pd.DataFrame): The dataset to generalize.
        quasi_identifiers (list): List of quasi-identifier columns.
        sensitive_attr (str): Sensitive attribute column name.
        l (int): ℓ-diversity threshold.
        max_levels (list): Maximum generalization levels for each quasi-identifier.
        hierarchies (list): Generalization hierarchies for each quasi-identifier.
        strategy (str): Search strategy ("bottom-up" or "top-down").

    Returns:
        pd.DataFrame: The generalized dataset satisfying ℓ-diversity.
    """
    print(Fore.CYAN + "🔍 Starting Lattice Search for Optimal ℓ-Diversity..." + Style.RESET_ALL)

    # Generate lattice levels
    lattice = generate_lattice(quasi_identifiers, max_levels)
    if strategy == "top-down":
        lattice = reversed(lattice)

    best_data = None
    best_utility = float("inf")  # Lower utility score = better

    for levels in lattice:
        print(Fore.YELLOW + f"Testing generalization levels: {levels}" + Style.RESET_ALL)

        # Apply generalization for the current lattice level
        generalized_data = apply_generalization(data, quasi_identifiers, levels, hierarchies)

        # Check ℓ-diversity for the generalized dataset
        if check_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l):
            utility_score = calculate_utility_score(generalized_data, quasi_identifiers, levels)
            print(Fore.GREEN + f"ℓ-Diversity satisfied at levels {levels} with utility score: {utility_score}" + Style.RESET_ALL)
        
            # Update the best data if this utility score is better
            if utility_score < best_utility:
                best_utility = utility_score
                best_data = generalized_data.copy()
                print(Fore.LIGHTGREEN_EX + f"✨ Found new optimal generalization: {levels}" + Style.RESET_ALL)

        else:
            print(Fore.RED + f"ℓ-Diversity not satisfied at levels {levels}" + Style.RESET_ALL)

    if best_data is not None:
        print(Fore.GREEN + f"✅ Optimal generalization found with utility score: {best_utility}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "❌ No generalization satisfying ℓ-diversity was found in the lattice." + Style.RESET_ALL)

    return best_data if best_data is not None else data

def redistribute_sensitive_values(data, quasi_identifiers, sensitive_attr, l, redistributed_records):
    """
    Redistribute sensitive values across groups to ensure ℓ-diversity.

    Args:
        data (pd.DataFrame): The dataset after generalization adjustments.
        quasi_identifiers (list): List of quasi-identifier columns.
        sensitive_attr (str): Sensitive attribute column name.
        l (int): ℓ-diversity threshold.
        redistributed_records (list): List to track redistributed record indices.

    Returns:
        bool: True if redistribution was successful, False otherwise.
    """
    groups = data.groupby(quasi_identifiers)
    success = False

    # Iterate over groups that fail ℓ-diversity
    for group_name, group in groups:
        sensitive_counts = group[sensitive_attr].value_counts()

        # Check if the group satisfies ℓ-diversity
        if len(sensitive_counts) < l:
            needed = l - len(sensitive_counts)  # Unique values needed
            print(Fore.RED + f"Group {group_name} fails ℓ-diversity. Needs {needed} additional unique sensitive values." + Style.RESET_ALL)

            # Search for sensitive values from other groups
            for other_group_name, other_group in groups:
                if group_name == other_group_name:
                    continue  # Skip the same group

                for value in other_group[sensitive_attr].unique():
                    if value not in sensitive_counts:
                        # Move the record containing this value
                        record_to_move = other_group[other_group[sensitive_attr] == value].iloc[0]
                        data.loc[record_to_move.name, quasi_identifiers] = list(group_name)  # Assign to the failing group
                        redistributed_records.append(record_to_move.name)  # Track redistributed record
                        sensitive_counts[value] = 1  # Add new value to the group
                        needed -= 1
                        success = True

                        print(Fore.GREEN + f"Redistributed record {record_to_move.name} with sensitive value '{value}' to group {group_name}." + Style.RESET_ALL)

                        if needed <= 0:
                            break

                if needed <= 0:
                    break

            # Log if redistribution was not fully successful
            if needed > 0:
                print(Fore.RED + f"Group {group_name} still fails ℓ-diversity after redistribution. Missing {needed} unique sensitive values." + Style.RESET_ALL)

    return success


def apply_l_diversity(data, quasi_identifiers, sensitive_attr, l, max_levels, hierarchies, redistributed_records=None, max_iterations=10):
    """
    Ensure ℓ-diversity by dynamically adjusting generalization levels and redistributing sensitive values.

    Args:
        data (pd.DataFrame): The k-anonymized dataset.
        quasi_identifiers (list): List of quasi-identifier columns.
        sensitive_attr (str): Sensitive attribute column name.
        l (int): ℓ-diversity threshold.
        max_levels (list): Maximum generalization levels for each quasi-identifier.
        hierarchies (list): Generalization hierarchies for each quasi-identifier.
        redistributed_records (list): List to track redistributed record indices.
        max_iterations (int): Maximum number of iterations to avoid infinite loops.

    Returns:
        pd.DataFrame: Refined dataset satisfying ℓ-diversity.
    """
    if redistributed_records is None:
        redistributed_records = []

    refined_data = data.copy()
    lattice = generate_lattice(quasi_identifiers, max_levels)
    print(Fore.YELLOW + f"\n🔍 Lattice Search Initialized: {len(lattice)} levels to test." + Style.RESET_ALL)

    for iteration in range(max_iterations):
        print(Fore.YELLOW + f"\nℓ-Diversity - Iteration {iteration + 1}" + Style.RESET_ALL)

        for levels in lattice:  # 🔍 Iterate through lattice levels dynamically
            print(Fore.BLUE + f"\n🔍 Testing generalization levels: {levels}" + Style.RESET_ALL)

            # Apply generalization for the current lattice level
            generalized_data = apply_generalization(refined_data, quasi_identifiers, levels, hierarchies)

            groups = generalized_data.groupby(quasi_identifiers)
            all_groups_satisfy = True  # Flag to check if all groups satisfy ℓ-diversity

            print(Fore.CYAN + "\n📌 Checking ℓ-Diversity Before Redistribution..." + Style.RESET_ALL)
            before_redistribution = {}

            for group_name, group in groups:
                sensitive_counts = group[sensitive_attr].value_counts()
                num_unique_sensitive_values = len(sensitive_counts)
                before_redistribution[group_name] = num_unique_sensitive_values

                if num_unique_sensitive_values >= l:
                    print(Fore.GREEN + f"✅ Group {group_name} satisfies ℓ-diversity: {num_unique_sensitive_values} unique sensitive values ≥ ℓ={l}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"❌ Group {group_name} fails ℓ-diversity: {num_unique_sensitive_values} unique sensitive values < ℓ={l}" + Style.RESET_ALL)
                    all_groups_satisfy = False

            # If generalization alone satisfies ℓ-diversity, stop here
            if all_groups_satisfy:
                print(Fore.GREEN + f"✅ ℓ-Diversity satisfied at levels {levels} without redistribution!" + Style.RESET_ALL)
                return generalized_data  # ✅ Return the successful generalization

            # 🔄 Perform Redistribution
            print(Fore.YELLOW + "🔄 Redistribution in progress..." + Style.RESET_ALL)
            success = redistribute_sensitive_values(generalized_data, quasi_identifiers, sensitive_attr, l, redistributed_records)

            # ✅ Recheck ℓ-Diversity AFTER Redistribution
            groups_after_redistribution = generalized_data.groupby(quasi_identifiers)

            print(Fore.CYAN + "\n📌 Checking ℓ-Diversity After Redistribution..." + Style.RESET_ALL)
            after_redistribution = {}

            all_groups_satisfy_after_redistribution = True  # Track ℓ-diversity status after redistribution

            for group_name, group in groups_after_redistribution:
                sensitive_counts = group[sensitive_attr].value_counts()
                num_unique_sensitive_values = len(sensitive_counts)
                after_redistribution[group_name] = num_unique_sensitive_values

                if num_unique_sensitive_values >= l:
                    print(Fore.GREEN + f"✅ Group {group_name} satisfies ℓ-diversity: {num_unique_sensitive_values} unique sensitive values ≥ ℓ={l}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"❌ Group {group_name} still fails ℓ-diversity: {num_unique_sensitive_values} unique sensitive values < ℓ={l}" + Style.RESET_ALL)
                    all_groups_satisfy_after_redistribution = False  # Mark failure

            # 🔄 Compare before and after redistribution
            print(Fore.MAGENTA + "\n🔍 Changes in Group Structures After Redistribution:" + Style.RESET_ALL)
            for group_name in before_redistribution:
                before_count = before_redistribution[group_name]
                after_count = after_redistribution.get(group_name, 0)  # Get after count, default to 0 if removed

                if before_count != after_count:
                    print(Fore.YELLOW + f"🔄 Group {group_name}: Before ℓ-Diversity = {before_count}, After = {after_count}" + Style.RESET_ALL)

            # ✅ If ℓ-diversity is satisfied after redistribution, stop further generalization
            if all_groups_satisfy_after_redistribution:
                print(Fore.GREEN + f"✅ ℓ-Diversity satisfied at levels {levels} after redistribution!" + Style.RESET_ALL)
                return generalized_data  # ✅ Return the refined dataset

        # Break the lattice loop if ℓ-diversity is satisfied
        print(Fore.RED + "ℓ-Diversity not satisfied at this iteration. Testing next levels in the lattice..." + Style.RESET_ALL)

    print(Fore.RED + "❌ No generalization satisfying ℓ-diversity was found in the lattice." + Style.RESET_ALL)
    return refined_data  # Return the most refined dataset found


def display_l_diverse_groups(k_anonymous_data, l_diverse_data, quasi_identifiers, sensitive_attr, redistributed_records, dataset_name="Dataset", l=3):
    """
    Display k-anonymous and ℓ-diverse dataset groups side-by-side for comparison, highlighting redistributed records
    and showing sensitive attribute counts for both groups.

    Args:
        k_anonymous_data (pd.DataFrame): The original k-anonymous dataset.
        l_diverse_data (pd.DataFrame): The dataset refined for ℓ-diversity.
        quasi_identifiers (list): List of quasi-identifier columns.
        sensitive_attr (str): Sensitive attribute column name.
        redistributed_records (list): List of redistributed record indices for tracking changes.
        dataset_name (str): Name of the dataset (default: "Dataset").
        l (int): ℓ-diversity threshold.
    """

    k_anonymous_groups = k_anonymous_data.groupby(quasi_identifiers)
    l_diverse_groups = l_diverse_data.groupby(quasi_identifiers)

    print(Fore.YELLOW + f"\n{dataset_name} - Comparison of k-Anonymized and ℓ-Diverse Groups:" + Style.RESET_ALL)

    total_distributed = 0

    for group, k_group_data in k_anonymous_groups:
        print(Fore.CYAN + f"\nGroup (Based on {', '.join(quasi_identifiers)}): {group}" + Style.RESET_ALL)

        # Display k-anonymous group
        print(Fore.BLUE + "k-Anonymized Group:" + Style.RESET_ALL)
        print(k_group_data.to_string(index=False))
        k_sensitive_counts = k_group_data[sensitive_attr].value_counts()
        print(Fore.LIGHTBLUE_EX + f"k-Anonymized Sensitive Attribute Counts: {k_sensitive_counts.to_dict()}" + Style.RESET_ALL)

        # Display ℓ-diverse group
        if group in l_diverse_groups.groups:
            l_group_data = l_diverse_groups.get_group(group)
            l_sensitive_counts = l_group_data[sensitive_attr].value_counts()
            num_unique_sensitive_values = len(l_sensitive_counts)

            print(Fore.GREEN + "ℓ-Diverse Group:" + Style.RESET_ALL)
            print(l_group_data.to_string(index=False))
            print(Fore.LIGHTGREEN_EX + f"ℓ-Diverse Sensitive Attribute Counts: {l_sensitive_counts.to_dict()}" + Style.RESET_ALL)

            # Check ℓ-diversity satisfaction
            if num_unique_sensitive_values >= l:
                print(Fore.GREEN + f"✅ Group satisfies ℓ-diversity: {num_unique_sensitive_values} unique sensitive values ≥ ℓ={l}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ Group fails ℓ-diversity: {num_unique_sensitive_values} unique sensitive values < ℓ={l}" + Style.RESET_ALL)

            # Highlight redistributed records
            redistributed_in_group = l_group_data.index.intersection(redistributed_records)
            redistributed_count = len(redistributed_in_group)
            total_distributed += redistributed_count

            if redistributed_count > 0:
                print(Fore.YELLOW + f"Redistributed Records in this Group: {redistributed_count}" + Style.RESET_ALL)
                print(l_group_data.loc[redistributed_in_group].to_string(index=False))
        else:
            print(Fore.RED + "ℓ-Diverse Group: No matching group found (group removed or restructured)." + Style.RESET_ALL)

        print(Fore.MAGENTA + "-" * 80 + Style.RESET_ALL)

    print(Fore.CYAN + f"\nTotal Redistributed Records Across All Groups: {total_distributed}" + Style.RESET_ALL)


def entropy(sensitive_counts):
    """Calculate the entropy of sensitive attribute values."""
    total = sum(sensitive_counts)
    probabilities = [count / total for count in sensitive_counts]
    return -sum(p * math.log(p, 2) for p in probabilities if p > 0)

def recursive_l_diversity(sensitive_counts, l):
    """Check recursive ℓ-diversity condition."""
    sorted_counts = sorted(sensitive_counts, reverse=True)
    if len(sorted_counts) < l:
        return False
    return sorted_counts[0] <= sum(sorted_counts[1:l])  # Most frequent ≤ sum of the next l-1 frequent

def apply_l_diversity_entropy_recursion(data, quasi_identifiers, sensitive_attr, l, max_levels, hierarchies, redistributed_records=None, max_iterations=10, diversity_type="basic"):
    """
    Extend ℓ-diversity to support entropy ℓ-diversity and recursive ℓ-diversity.

    Args:
        data (pd.DataFrame): The k-anonymized dataset.
        quasi_identifiers (list): List of quasi-identifier columns.
        sensitive_attr (str): Sensitive attribute column name.
        l (int): ℓ-diversity threshold.
        max_levels (list): Maximum generalization levels for each quasi-identifier.
        hierarchies (list): Generalization hierarchies for each quasi-identifier.
        redistributed_records (list): List to track redistributed record indices.
        max_iterations (int): Maximum number of iterations to avoid infinite loops.
        diversity_type (str): Type of ℓ-diversity - "basic", "entropy", or "recursive".

    Returns:
        pd.DataFrame: Refined dataset satisfying the chosen ℓ-diversity.
    """
    if redistributed_records is None:
        redistributed_records = []

    refined_data = data.copy()
    lattice = generate_lattice(quasi_identifiers, max_levels)
    print(Fore.YELLOW + f"\n🔍 Lattice Search Initialized: {len(lattice)} levels to test." + Style.RESET_ALL)

    for iteration in range(max_iterations):
        print(Fore.YELLOW + f"\nℓ-Diversity - Iteration {iteration + 1}" + Style.RESET_ALL)

        for levels in lattice:
            print(Fore.BLUE + f"\n🔍 Testing generalization levels: {levels}" + Style.RESET_ALL)

            # Apply generalization for the current lattice level
            generalized_data = apply_generalization(refined_data, quasi_identifiers, levels, hierarchies)

            print(Fore.CYAN + "\n📌 Checking ℓ-Diversity Before Redistribution..." + Style.RESET_ALL)
            groups = generalized_data.groupby(quasi_identifiers)
            all_groups_satisfy = True  # Flag to check if all groups satisfy ℓ-diversity
            before_redistribution = {}

            for group_name, group in groups:
                sensitive_counts = group[sensitive_attr].value_counts().values
                num_unique_sensitive_values = len(sensitive_counts)

                # ℓ-Diversity Calculations
                if diversity_type == "basic":
                    satisfies = num_unique_sensitive_values >= l
                    log_message = f"{num_unique_sensitive_values} unique values {'≥' if satisfies else '<'} ℓ={l}"
                elif diversity_type == "entropy":
                    calculated_entropy = entropy(sensitive_counts)
                    threshold = math.log(l, 2)
                    satisfies = calculated_entropy >= threshold
                    log_message = f"Entropy = {calculated_entropy:.4f} {'≥' if satisfies else '<'} log(ℓ) = {threshold:.4f}"
                elif diversity_type == "recursive":
                    sorted_counts = sorted(sensitive_counts, reverse=True)
                    if len(sorted_counts) < l:
                        satisfies = False
                        log_message = f"Not enough unique values ({len(sorted_counts)} < ℓ={l})"
                    else:
                        most_frequent = sorted_counts[0]
                        next_l_minus_1 = sum(sorted_counts[1:l])
                        satisfies = most_frequent <= next_l_minus_1
                        log_message = f"Most frequent = {most_frequent}, Sum of next {l-1} = {next_l_minus_1} -> {'≤' if satisfies else '>'}"

                # Log the result
                if satisfies:
                    print(Fore.GREEN + f"✅ Group {group_name} satisfies {diversity_type} ℓ-diversity: {log_message}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"❌ Group {group_name} fails {diversity_type} ℓ-diversity: {log_message}" + Style.RESET_ALL)
                    all_groups_satisfy = False

                before_redistribution[group_name] = satisfies  # Track satisfaction status before redistribution

            # If all groups satisfy ℓ-diversity before redistribution
            if all_groups_satisfy:
                print(Fore.GREEN + f"✅ ℓ-Diversity satisfied at levels {levels} without redistribution!" + Style.RESET_ALL)
                return generalized_data

            # Perform Redistribution
            print(Fore.YELLOW + "🔄 Redistribution in progress..." + Style.RESET_ALL)
            redistribute_sensitive_values(generalized_data, quasi_identifiers, sensitive_attr, l, redistributed_records)

            # Recheck ℓ-Diversity after redistribution
            print(Fore.CYAN + "\n📌 Checking ℓ-Diversity After Redistribution..." + Style.RESET_ALL)
            groups_after_redistribution = generalized_data.groupby(quasi_identifiers)
            all_groups_satisfy_after_redistribution = True
            after_redistribution = {}

            for group_name, group in groups_after_redistribution:
                sensitive_counts = group[sensitive_attr].value_counts().values
                num_unique_sensitive_values = len(sensitive_counts)

                # ℓ-Diversity Calculations After Redistribution
                if diversity_type == "basic":
                    satisfies = num_unique_sensitive_values >= l
                    log_message = f"{num_unique_sensitive_values} unique values {'≥' if satisfies else '<'} ℓ={l}"
                elif diversity_type == "entropy":
                    calculated_entropy = entropy(sensitive_counts)
                    threshold = math.log(l, 2)
                    satisfies = calculated_entropy >= threshold
                    log_message = f"Entropy = {calculated_entropy:.4f} {'≥' if satisfies else '<'} log(ℓ) = {threshold:.4f}"
                elif diversity_type == "recursive":
                    sorted_counts = sorted(sensitive_counts, reverse=True)
                    if len(sorted_counts) < l:
                        satisfies = False
                        log_message = f"Not enough unique values ({len(sorted_counts)} < ℓ={l})"
                    else:
                        most_frequent = sorted_counts[0]
                        next_l_minus_1 = sum(sorted_counts[1:l])
                        satisfies = most_frequent <= next_l_minus_1
                        log_message = f"Most frequent = {most_frequent}, Sum of next {l-1} = {next_l_minus_1} -> {'≤' if satisfies else '>'}"

                # Log the result
                if satisfies:
                    print(Fore.GREEN + f"✅ Group {group_name} satisfies {diversity_type} ℓ-diversity: {log_message}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + f"❌ Group {group_name} fails {diversity_type} ℓ-diversity: {log_message}" + Style.RESET_ALL)
                    all_groups_satisfy_after_redistribution = False

                after_redistribution[group_name] = satisfies  # Track satisfaction status after redistribution

            # Compare results before and after redistribution
            print(Fore.MAGENTA + "\n🔍 Changes in ℓ-Diversity Satisfaction After Redistribution:" + Style.RESET_ALL)
            for group_name, before_status in before_redistribution.items():
                after_status = after_redistribution.get(group_name, False)
                if before_status != after_status:
                    print(Fore.YELLOW + f"🔄 Group {group_name}: Before = {'Satisfied' if before_status else 'Failed'}, After = {'Satisfied' if after_status else 'Failed'}" + Style.RESET_ALL)

            # If ℓ-Diversity satisfied after redistribution
            if all_groups_satisfy_after_redistribution:
                print(Fore.GREEN + f"✅ ℓ-Diversity satisfied at levels {levels} after redistribution!" + Style.RESET_ALL)
                return generalized_data

        print(Fore.RED + f"ℓ-Diversity not satisfied at iteration {iteration + 1}. Testing next levels in the lattice..." + Style.RESET_ALL)

    print(Fore.RED + f"❌ No generalization satisfying {diversity_type} ℓ-diversity was found in the lattice." + Style.RESET_ALL)
    return refined_data  # Return the most refined dataset found



if __name__ == "__main__":
    # File paths for raw datasets
    adult_data_path = "./data/k_anonymity_dataset/adult_k_anonymized.csv"
    lands_end_data_path = "./data/k_anonymity_dataset/lands_end_k_anonymized.csv"

    # Ensure raw datasets exist
    if not os.path.exists(adult_data_path) or not os.path.exists(lands_end_data_path):
        print(Fore.RED + "Required raw datasets are missing." + Style.RESET_ALL)
        print(Fore.YELLOW + "Please ensure the following files exist:" + Style.RESET_ALL)
        print(f"  - {adult_data_path}")
        print(f"  - {lands_end_data_path}")
        exit(1)

    # Load raw datasets
    try:
        adult_df = pd.read_csv(adult_data_path)
        lands_end_df = pd.read_csv(lands_end_data_path)
        print(Fore.GREEN + "Raw datasets loaded successfully." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error loading datasets: {e}" + Style.RESET_ALL)
        exit(1)

    # Parameters for Adult Dataset
    adult_params = {
        "max_levels": [4, 1, 1, 2],  # Generalization levels for quasi-identifiers
        "hierarchies": ["range", "suppression", "taxonomy", "taxonomy"],
    }
    adult_quasi_identifiers = ["Age", "Gender", "Education", "Work Class"]
    adult_sensitive_attr = "Occupation"

    # Define l as a variable
    l = 12  # ℓ-diversity threshold

    # Step 1: Apply Generalization for k-Anonymity
    print(Fore.BLUE + "Applying k-Anonymity for Adult Dataset..." + Style.RESET_ALL)
    generalized_data = apply_generalization(
        data=adult_df,
        quasi_identifiers=adult_quasi_identifiers,
        levels=[1, 0, 1, 1],  # Initial generalization levels
        hierarchies=adult_params["hierarchies"]
    )

    # Check k-Anonymity
    k = 4
    if not check_k_anonymity(generalized_data, adult_quasi_identifiers, k):
        print(Fore.RED + "Dataset does not satisfy k-anonymity. Exiting..." + Style.RESET_ALL)
        exit(1)

    redistributed_records = []

    # # Step 2: Apply ℓ-Diversity for the Adult Dataset
    # divider = Fore.LIGHTBLACK_EX + "-" * 80 + Style.RESET_ALL
    # print(divider)
    # print(Fore.BLUE + "Applying ℓ-Diversity for Adult Dataset..." + Style.RESET_ALL)
    # l_diverse_data = apply_l_diversity(
    #     data=generalized_data,  # Generalized (k-anonymized) dataset
    #     quasi_identifiers=adult_quasi_identifiers,  # List of QIs
    #     sensitive_attr=adult_sensitive_attr,  # Sensitive attribute
    #     l=l,  # ℓ-diversity threshold
    #     max_levels=adult_params["max_levels"],  # Maximum generalization levels
    #     hierarchies=adult_params["hierarchies"],  # Hierarchies for generalization
    #     redistributed_records=redistributed_records  # Track redistributed records
    # )

    # Save ℓ-Diverse Data
    # output_path = "./data/l_diversity_dataset/adult_l_diverse.csv"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # l_diverse_data.to_csv(output_path, index=False)
    # print(Fore.GREEN + f"ℓ-Diversity applied successfully to the Adult Dataset. Results saved to '{output_path}'." + Style.RESET_ALL)

    l_diverse_data = apply_l_diversity_entropy_recursion(
        data=generalized_data,
        quasi_identifiers=adult_quasi_identifiers,
        sensitive_attr="Occupation",
        l=3,
        max_levels=adult_params["max_levels"],
        hierarchies=adult_params["hierarchies"],
        diversity_type="entropy"
    )


    # Step 3: Display Comparison of k-Anonymized and ℓ-Diverse Groups
    # display_l_diverse_groups(
    #     k_anonymous_data=generalized_data,
    #     l_diverse_data=l_diverse_data,
    #     quasi_identifiers=adult_quasi_identifiers,
    #     sensitive_attr=adult_sensitive_attr,
    #     redistributed_records=redistributed_records,
    #     dataset_name="Adult Dataset",
    #     l = l
    # )
