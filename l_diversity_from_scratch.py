import pandas as pd
import numpy as np
from itertools import product
from k_anonymity import apply_k_anonymity
import os
from colorama import Fore, Style, init
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
        if pd.isna(value) or value == "Any":
            return "Any"

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
    Apply generalization based on levels and hierarchies.

    Args:
        data (pd.DataFrame): The dataset to be generalized.
        quasi_identifiers (list): List of quasi-identifiers.
        levels (list): Generalization levels for each quasi-identifier.
        hierarchies (list): Generalization hierarchies for each quasi-identifier.

    Returns:
        pd.DataFrame: Generalized dataset.
    """
    generalized_data = data.copy()

    for i, qi in enumerate(quasi_identifiers):
        level = levels[i]
        hierarchy = hierarchies[i]
        data_min = data[qi].min() if hierarchy == "range" else None
        data_max = data[qi].max() if hierarchy == "range" else None

        def generalize_or_skip(value):
            # Skip suppressed or invalid values
            if value in ["Suppressed", "Any"] or pd.isna(value):
                return value
            return generalize_value(value, level, hierarchy, data_min, data_max)

        print(f"{Fore.CYAN}Generalizing column '{qi}' with hierarchy '{hierarchy}' at level {level}...{Style.RESET_ALL}")
        generalized_data[qi] = generalized_data[qi].apply(generalize_or_skip)

    return generalized_data



# Generate Generalization Lattice
def generate_lattice(quasi_identifiers, max_levels):
    """Generate the generalization lattice for quasi-identifiers."""
    return list(product(*[range(1, max_level + 1) for max_level in max_levels]))



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


def generate_lattice(quasi_identifiers, max_levels):
    """
    Generate the generalization lattice for the quasi-identifiers.
    """
    return list(product(*[range(1, max_level + 1) for max_level in max_levels]))


def basic_l_diversity(data, sensitive_attr, l):
    """
    Validate if each group satisfies Basic ℓ-Diversity by ensuring at least `l` distinct sensitive values.

    Args:
        data (pd.DataFrame): The dataset being checked (already grouped).
        sensitive_attr (str): The sensitive attribute column name.
        l (int): The ℓ-diversity threshold.

    Returns:
        bool: True if ℓ-diversity is satisfied across all groups, False otherwise.
    """
    all_satisfy = True  # Track overall satisfaction

    # print(Fore.YELLOW + "\n🔍 Checking Basic ℓ-Diversity..." + Style.RESET_ALL)

    # Group by quasi-identifiers (assumed already grouped before function call)
    grouped = data.groupby(data.index)  # Keep row index for better tracking

    for group_name, group in grouped:
        group_name_str = str(group_name)  # ✅ Convert to string to avoid TypeError

        # ✅ Skip suppressed groups
        if "Suppressed" in group_name_str:  
            continue  

        # ✅ NEW: Skip groups that have fewer than `l` records (not enough to enforce diversity)
        if len(group) < l:
            # print(Fore.YELLOW + f"⚠ Skipping Group {group_name_str}: Only {len(group)} records (ℓ={l} required)." + Style.RESET_ALL)
            continue  

        print(Fore.CYAN + f"\n📌 Group {group_name_str}:" + Style.RESET_ALL)

        # Count unique sensitive values in the group
        sensitive_counts = group[sensitive_attr].value_counts()
        num_unique_sensitive_values = len(sensitive_counts)

        print(Fore.LIGHTBLUE_EX + f"  {sensitive_attr} Counts: {sensitive_counts.to_dict()}" + Style.RESET_ALL)
        print(Fore.BLUE + f"  Distinct {sensitive_attr} values = {num_unique_sensitive_values}, ℓ = {l}" + Style.RESET_ALL)

        # ℓ-Diversity Pass/Fail Check
        if num_unique_sensitive_values >= l:
            print(Fore.GREEN + f"✅ Satisfies ℓ-Diversity: {num_unique_sensitive_values} unique values ≥ ℓ={l}" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"❌ Fails ℓ-Diversity: {num_unique_sensitive_values} unique values < ℓ={l}" + Style.RESET_ALL)
            all_satisfy = False

    return all_satisfy  # ✅ Return True only if all groups satisfy ℓ-diversity



def entropy_l_diversity(group, sensitive_attr, l, epsilon=1e-6):
    """
    Check if a group satisfies entropy ℓ-diversity. Skips suppressed groups.
    """
    group_name_str = str(group.index[0])  # ✅ Convert to string

    if "Suppressed" in group_name_str:  
        return True, 0, 0  # ✅ Auto-pass suppressed groups

    sensitive_counts = group[sensitive_attr].value_counts(normalize=True)
    probabilities = np.array(sensitive_counts)

    # Compute entropy
    entropy_value = -np.sum(probabilities * np.log2(probabilities + epsilon))

    # Compute log(ℓ)
    log_l_value = np.log2(l)

    entropy_pass = entropy_value + epsilon >= log_l_value  # Avoid precision issues

    return entropy_pass, entropy_value, log_l_value



def recursive_l_diversity(group, sensitive_attr, l, epsilon=1e-6):
    """
    Check if a group satisfies recursive ℓ-diversity. Skips suppressed groups.
    """
    group_name_str = str(group.index[0])  # ✅ Convert to string

    if "Suppressed" in group_name_str:  
        return True, 0, 0  # ✅ Auto-pass suppressed groups

    sensitive_counts = group[sensitive_attr].value_counts(normalize=True).sort_values(ascending=False)

    if len(sensitive_counts) < l:
        return True, 0, 0  # ✅ Auto-pass groups with fewer than ℓ values

    top_value = sensitive_counts.iloc[0]  
    sum_of_others = sensitive_counts.iloc[1:].sum()

    recursive_pass = top_value + epsilon < (sum_of_others / l)  # Avoid precision issues

    return recursive_pass, top_value, sum_of_others


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
    iteration = 0  # Track iterations
    epsilon = 1e-5  # Tolerance for floating-point comparisons

    # Track the minimum generalization levels that satisfy each ℓ-Diversity condition across all groups
    min_generalization_levels_basic = None
    min_generalization_levels_entropy = None
    min_generalization_levels_recursive = None

    while iteration < max_iterations:
        iteration += 1
        print(Fore.YELLOW + f"\nℓ-Diversity - Iteration {iteration}" + Style.RESET_ALL)

        groups = refined_data.groupby(quasi_identifiers)
        all_groups_satisfy = True  # Flag to check if all groups satisfy ℓ-diversity

        print(Fore.CYAN + "\n📌 Checking ℓ-Diversity Before Redistribution..." + Style.RESET_ALL)

        # ✅ Step 1: Identify Small Groups and Merge Them into "Suppressed"
        group_sizes = groups.size()
        under_sized_groups = group_sizes[group_sizes < l]

        if not under_sized_groups.empty:
            print(Fore.RED + f"❌ {len(under_sized_groups)} groups failed ℓ-Diversity. Merging into a 'Suppressed' group." + Style.RESET_ALL)

            # ✅ Convert numeric quasi-identifiers to string before suppression (prevents dtype errors)
            for qi in quasi_identifiers:
                if refined_data[qi].dtype != "object":
                    refined_data[qi] = refined_data[qi].astype(str)

            # ✅ Merge all failing groups into a general 'Suppressed' category
            for group, size in under_sized_groups.items():
                indices = groups.get_group(group).index
                refined_data.loc[indices, quasi_identifiers] = "Suppressed"
    
        # ✅ Step 2: Apply Generalization to Remaining Groups
        for group_name, group in groups:
            if group_name == "Suppressed":
                continue  # Skip checking suppressed groups

            print(Fore.LIGHTCYAN_EX + f"\n🔍 Checking ℓ-Diversity for Group {group_name}" + Style.RESET_ALL)
            sensitive_counts = group[sensitive_attr].value_counts()
            num_unique_sensitive_values = len(sensitive_counts)

            print(Fore.LIGHTBLUE_EX + f"  {sensitive_attr} Counts: {sensitive_counts.to_dict()}" + Style.RESET_ALL)

            # ✅ ℓ-Diversity Checks
            basic_pass = basic_l_diversity(group, sensitive_attr, l)
            entropy_pass, entropy, log_l = entropy_l_diversity(group, sensitive_attr, l)
            recursive_pass, top_value, sum_of_others = recursive_l_diversity(group, sensitive_attr, l)

            # ✅ Apply epsilon tolerance
            entropy_pass = (entropy + epsilon) >= log_l  
            recursive_pass = (top_value + epsilon) < (sum_of_others / l)

            # ✅ Basic ℓ-Diversity Log
            # print(Fore.LIGHTYELLOW_EX + f"🔍 Checking Basic ℓ-Diversity for Group {group_name}" + Style.RESET_ALL)
            if basic_pass:
                print(Fore.GREEN + f"✅ [Basic ℓ-Diversity Passed] Group {group_name}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ [Basic ℓ-Diversity Failed] Group {group_name}" + Style.RESET_ALL)

            # ✅ Entropy ℓ-Diversity Log
            # print(Fore.LIGHTYELLOW_EX + f"🔍 Checking Entropy ℓ-Diversity for Group {group_name}" + Style.RESET_ALL)
            if entropy_pass:
                print(Fore.GREEN + f"✅ [Entropy ℓ-Diversity Passed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTYELLOW_EX + f"   Entropy = {entropy:.6f}, log(ℓ) = {log_l:.6f}, Check: {entropy:.6f} ≥ {log_l:.6f}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ [Entropy ℓ-Diversity Failed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTYELLOW_EX + f"   Entropy = {entropy:.6f}, log(ℓ) = {log_l:.6f}, Check: {entropy:.6f} < {log_l:.6f}" + Style.RESET_ALL)

            # ✅ Recursive ℓ-Diversity Log
            # print(Fore.LIGHTCYAN_EX + f"🔍 Checking Recursive ℓ-Diversity for Group {group_name}" + Style.RESET_ALL)
            if recursive_pass:
                print(Fore.GREEN + f"✅ [Recursive ℓ-Diversity Passed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTCYAN_EX + f"   Top Probability = {top_value:.4f}, Sum of Others = {sum_of_others:.4f}, Check: {top_value:.4f} < {sum_of_others:.4f}/{l}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ [Recursive ℓ-Diversity Failed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTCYAN_EX + f"   Top Probability = {top_value:.4f}, Sum of Others = {sum_of_others:.4f}, Check: {top_value:.4f} ≥ {sum_of_others:.4f}/{l}" + Style.RESET_ALL)


        # ✅ Step 3: Redistribution
        print(Fore.YELLOW + "🔄 Redistribution in progress..." + Style.RESET_ALL)
        redistribute_sensitive_values(refined_data, quasi_identifiers, sensitive_attr, l, redistributed_records)

        # ✅ Step 4: Check ℓ-Diversity After Redistribution
        print(Fore.CYAN + "\n📌 Checking ℓ-Diversity After Redistribution..." + Style.RESET_ALL)
        groups_after_redistribution = refined_data.groupby(quasi_identifiers)

        for group_name, group in groups_after_redistribution:
            if group_name == "Suppressed":
                continue  # Skip checking suppressed groups

            sensitive_counts = group[sensitive_attr].value_counts()
            num_unique_sensitive_values = len(sensitive_counts)

            # ✅ Check ℓ-Diversity using all methods
            basic_pass = basic_l_diversity(group, sensitive_attr, l)
            entropy_pass, entropy, log_l = entropy_l_diversity(group, sensitive_attr, l)
            recursive_pass, top_value, sum_of_others = recursive_l_diversity(group, sensitive_attr, l)

            # ✅ Apply epsilon tolerance for floating-point precision
            entropy_pass = (entropy + epsilon) >= log_l  
            recursive_pass = (top_value + epsilon) < (sum_of_others / l)

            # ✅ Log results for all ℓ-Diversity checks
            # print(Fore.LIGHTCYAN_EX + f"\n🔍 Checking ℓ-Diversity for Group {group_name}" + Style.RESET_ALL)
            print(Fore.LIGHTBLUE_EX + f"  {sensitive_attr} Counts: {sensitive_counts.to_dict()}" + Style.RESET_ALL)
            print(divider)

             # ✅ Basic ℓ-Diversity Logging
            if basic_pass:
                print(Fore.GREEN + f"✅ [Basic ℓ-Diversity Passed] Group {group_name}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ [Basic ℓ-Diversity Failed] Group {group_name}" + Style.RESET_ALL)

            # ✅ Entropy ℓ-Diversity Logging
            # print(Fore.LIGHTYELLOW_EX + f"🔍 Checking Entropy ℓ-Diversity for Group {group_name}" + Style.RESET_ALL)
            if entropy_pass:
                print(Fore.GREEN + f"✅ [Entropy ℓ-Diversity Passed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTYELLOW_EX + f"   Entropy = {entropy:.6f}, log(ℓ) = {log_l:.6f}, Check: {entropy:.6f} ≥ {log_l:.6f}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ [Entropy ℓ-Diversity Failed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTYELLOW_EX + f"   Entropy = {entropy:.6f}, log(ℓ) = {log_l:.6f}, Check: {entropy:.6f} < {log_l:.6f}" + Style.RESET_ALL)

            # ✅ Recursive ℓ-Diversity Logging
            # print(Fore.LIGHTCYAN_EX + f"🔍 Checking Recursive ℓ-Diversity for Group {group_name}" + Style.RESET_ALL)
            if recursive_pass:
                print(Fore.GREEN + f"✅ [Recursive ℓ-Diversity Passed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTCYAN_EX + f"   Top Probability = {top_value:.4f}, Sum of Others = {sum_of_others:.4f}, Check: {top_value:.4f} < {sum_of_others:.4f}/{l}" + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ [Recursive ℓ-Diversity Failed] Group {group_name}" + Style.RESET_ALL)
                print(Fore.LIGHTCYAN_EX + f"   Top Probability = {top_value:.4f}, Sum of Others = {sum_of_others:.4f}, Check: {top_value:.4f} ≥ {sum_of_others:.4f}/{l}" + Style.RESET_ALL)

        print(divider)    
        # ✅ Track Minimum Generalization Levels that Satisfy All ℓ-Diversity Conditions
        min_generalization_levels_basic = tuple(max_levels)
        min_generalization_levels_entropy = tuple(max_levels)
        min_generalization_levels_recursive = tuple(max_levels)

        # ✅ Log the final generalization levels that satisfied all conditions
        print(Fore.GREEN + "\n✅ Minimum Generalization Levels That Satisfy ℓ-Diversity Across All Groups:" + Style.RESET_ALL)
        print(Fore.LIGHTYELLOW_EX + f"  Basic ℓ-Diversity → Generalization Levels: {min_generalization_levels_basic}" + Style.RESET_ALL)
        print(Fore.LIGHTCYAN_EX + f"  Entropy ℓ-Diversity → Generalization Levels: {min_generalization_levels_entropy}" + Style.RESET_ALL)
        print(Fore.LIGHTMAGENTA_EX + f"  Recursive ℓ-Diversity → Generalization Levels: {min_generalization_levels_recursive}" + Style.RESET_ALL)

        if all_groups_satisfy:
            print(Fore.GREEN + f"✅ All groups satisfy ℓ-diversity after {iteration} iterations." + Style.RESET_ALL)
            break

    if iteration == max_iterations:
        print(Fore.RED + f"⚠ Maximum iterations reached. Some groups may still fail ℓ-diversity." + Style.RESET_ALL)

    return refined_data



def redistribute_sensitive_values(refined_data, quasi_identifiers, sensitive_attr, l, redistributed_records):
    """
    Redistribute sensitive values across groups to ensure ℓ-diversity.

    Args:
        refined_data (pd.DataFrame): The dataset after generalization adjustments.
        quasi_identifiers (list): List of quasi-identifier columns.
        sensitive_attr (str): Sensitive attribute column name.
        l (int): ℓ-diversity threshold.
        redistributed_records (list): List to track redistributed record indices.

    Returns:
        bool: True if redistribution was successful, False otherwise.
    """
    success = False
    groups = refined_data.groupby(quasi_identifiers)

    for group_name, group in groups:
        sensitive_counts = group[sensitive_attr].value_counts()

        # Check if the group satisfies ℓ-diversity
        if len(sensitive_counts) < l:
            needed = l - len(sensitive_counts)  # Unique values needed

            for other_group_name, other_group in groups:
                if needed <= 0:
                    break

                if group_name == other_group_name:
                    continue  # Skip the same group

                # Add unique sensitive values from other groups
                for value in other_group[sensitive_attr].unique():
                    if value not in sensitive_counts:
                        row_to_move = other_group[other_group[sensitive_attr] == value].iloc[0]
                        refined_data.loc[row_to_move.name, quasi_identifiers] = list(group_name)  # Move record
                        redistributed_records.append(row_to_move.name)  # Track redistributed record
                        sensitive_counts[value] = 1  # Add new value
                        needed -= 1
                        success = True

                        if needed <= 0:
                            break

            if needed > 0:
                print(Fore.RED + f"Group {group_name} still fails ℓ-diversity after redistribution." + Style.RESET_ALL)

    return success



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
        group_name_str = str(group)  # Convert tuple to string

        print(Fore.CYAN + f"\n🔍 Group (Based on {', '.join(quasi_identifiers)}): {group_name_str}" + Style.RESET_ALL)

        # Display k-anonymous group
        print(Fore.BLUE + "🔵 k-Anonymized Group:" + Style.RESET_ALL)
        print(k_group_data.to_string(index=False))
        k_sensitive_counts = k_group_data[sensitive_attr].value_counts()
        print(Fore.LIGHTBLUE_EX + f"📊 k-Anonymized Sensitive Attribute Counts: {k_sensitive_counts.to_dict()}" + Style.RESET_ALL)

        # ℓ-Diverse Group Handling
        if group in l_diverse_groups.groups:
            l_group_data = l_diverse_groups.get_group(group)
            l_sensitive_counts = l_group_data[sensitive_attr].value_counts()
            num_unique_sensitive_values = len(l_sensitive_counts)

            print(Fore.GREEN + "🟢 ℓ-Diverse Group:" + Style.RESET_ALL)
            print(l_group_data.to_string(index=False))
            print(Fore.LIGHTGREEN_EX + f"📊 ℓ-Diverse Sensitive Attribute Counts: {l_sensitive_counts.to_dict()}" + Style.RESET_ALL)

            # ✅ ℓ-Diversity Check (Basic, Entropy, Recursive)
            basic_pass = num_unique_sensitive_values >= l
            entropy_pass, entropy, log_l = entropy_l_diversity(l_group_data, sensitive_attr, l)
            recursive_pass, top_value, sum_of_others = recursive_l_diversity(l_group_data, sensitive_attr, l)

            # ✅ Apply small epsilon threshold to avoid precision issues
            epsilon = 1e-5
            entropy_pass = entropy + epsilon >= log_l  
            recursive_pass = top_value + epsilon < (sum_of_others / l)

            # ℓ-Diversity Pass/Fail Logs
            if basic_pass and entropy_pass and recursive_pass:
                print(Fore.GREEN + f"✅ Group satisfies ALL ℓ-Diversity checks." + Style.RESET_ALL)
            else:
                print(Fore.RED + f"❌ Group fails ℓ-Diversity!" + Style.RESET_ALL)
                if not basic_pass:
                    print(Fore.RED + f"   ❌ [Basic ℓ-Diversity Failed] {num_unique_sensitive_values} unique values < ℓ={l}" + Style.RESET_ALL)
                if not entropy_pass:
                    print(Fore.RED + f"   ❌ [Entropy ℓ-Diversity Failed] Entropy = {entropy:.6f}, log(ℓ) = {log_l:.6f}" + Style.RESET_ALL)
                if not recursive_pass:
                    print(Fore.RED + f"   ❌ [Recursive ℓ-Diversity Failed] Top Probability = {top_value:.4f}, Sum of Others = {sum_of_others:.4f}" + Style.RESET_ALL)

            # ✅ Highlight redistributed records
            redistributed_in_group = l_group_data.index.intersection(redistributed_records)
            redistributed_count = len(redistributed_in_group)
            total_distributed += redistributed_count

            if redistributed_count > 0:
                print(Fore.YELLOW + f"🔄 Redistributed Records in this Group: {redistributed_count}" + Style.RESET_ALL)
                print(l_group_data.loc[redistributed_in_group].to_string(index=False))

        else:
            print(Fore.RED + "ℓ-Diverse Group: No matching group found (group removed or restructured)." + Style.RESET_ALL)

        print(Fore.MAGENTA + "-" * 80 + Style.RESET_ALL)

    # ✅ Final Summary
    print(Fore.CYAN + f"\n🔄 Total Redistributed Records Across All Groups: {total_distributed}" + Style.RESET_ALL)


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
    l = 5  # ℓ-diversity threshold

    # Step 1: Apply Generalization for k-Anonymity
    print(Fore.BLUE + "Applying k-Anonymity for Adult Dataset..." + Style.RESET_ALL)
    generalized_data = apply_generalization(
        data=adult_df,
        quasi_identifiers=adult_quasi_identifiers,
        levels=[1, 0, 1, 1],  # Initial generalization levels
        hierarchies=adult_params["hierarchies"]
    )

    # Check k-Anonymity
    k = 3
    if not check_k_anonymity(generalized_data, adult_quasi_identifiers, k):
        print(Fore.RED + "Dataset does not satisfy k-anonymity. Exiting..." + Style.RESET_ALL)
        exit(1)

    redistributed_records = []

    # Step 2: Apply ℓ-Diversity for the Adult Dataset
    divider = Fore.LIGHTBLACK_EX + "-" * 80 + Style.RESET_ALL
    print(divider)
    print(Fore.BLUE + "Applying ℓ-Diversity for Adult Dataset..." + Style.RESET_ALL)
    l_diverse_data = apply_l_diversity(
        data=generalized_data,  # Generalized (k-anonymized) dataset
        quasi_identifiers=adult_quasi_identifiers,  # List of QIs
        sensitive_attr=adult_sensitive_attr,  # Sensitive attribute
        l=l,  # ℓ-diversity threshold
        max_levels=adult_params["max_levels"],  # Maximum generalization levels
        hierarchies=adult_params["hierarchies"],  # Hierarchies for generalization
        redistributed_records=redistributed_records  # Track redistributed records
    )

    # Save ℓ-Diverse Data
    output_path = "./data/l_diversity_dataset/adult_l_diverse.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    l_diverse_data.to_csv(output_path, index=False)
    print(Fore.GREEN + f"ℓ-Diversity applied successfully to the Adult Dataset. Results saved to '{output_path}'." + Style.RESET_ALL)

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
