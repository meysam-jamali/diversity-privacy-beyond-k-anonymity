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
        # Skip suppressed or invalid values
        if value == "Suppressed" or pd.isna(value) or value == "Any":
            return "Any"

        # Range-based generalization
        if hierarchy == "range":
            # Ensure numeric value
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"Non-numeric value '{value}' cannot be generalized with a 'range' hierarchy.")
            if data_min is None or data_max is None:
                raise ValueError("Range hierarchy requires data_min and data_max.")
            step = (data_max - data_min) / (10 ** level)
            lower_bound = (value // step) * step
            upper_bound = lower_bound + step
            return f"{int(lower_bound)}-{int(upper_bound)}"

        # Rounding-based generalization
        elif hierarchy == "round":
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"Non-numeric value '{value}' cannot be generalized with a 'round' hierarchy.")
            step = 10 ** level
            lower_bound = (value // step) * step
            upper_bound = lower_bound + step - 1
            return f"{int(lower_bound)}-{int(upper_bound)}"

        # Taxonomy-based generalization
        elif hierarchy == "taxonomy":
            if level == 1:
                return "General"
            elif level > 1:
                return f"Level-{level}"
            return value

        # Suppression-based generalization
        elif hierarchy == "suppression":
            return "Any" if level > 0 else value

        # Default case for unsupported hierarchies
        return value

    except ValueError as e:
        print(f"{Fore.RED}Error generalizing value '{value}' for hierarchy '{hierarchy}': {e}{Style.RESET_ALL}")
        return value  # Return the original value if an error occurs
    except Exception as e:
        print(f"{Fore.RED}Unexpected error while generalizing value '{value}': {e}{Style.RESET_ALL}")
        return value

def apply_generalization(data, quasi_identifiers, levels, max_levels, hierarchies):
    """
    Apply generalization based on levels, max_levels, and hierarchies.

    Args:
        data (pd.DataFrame): The dataset to be generalized.
        quasi_identifiers (list): List of quasi-identifiers.
        levels (list): Generalization levels for each quasi-identifier.
        max_levels (list): Maximum allowed levels for each quasi-identifier.
        hierarchies (list): Generalization hierarchies for each quasi-identifier.

    Returns:
        pd.DataFrame: Generalized dataset.
    """
    generalized_data = data.copy()

    for i, qi in enumerate(quasi_identifiers):
        level = levels[i]
        max_level = max_levels[i]
        hierarchy = hierarchies[i]

        # Skip columns that are already fully generalized or exceed max levels
        if level > max_level:
            print(f"{Fore.YELLOW}Skipping column '{qi}': Level {level} exceeds max allowed {max_level}.{Style.RESET_ALL}")
            continue

        print(f"{Fore.CYAN}Generalizing column '{qi}' with hierarchy '{hierarchy}' at level {level}...{Style.RESET_ALL}")

        # Handle range generalization and skip non-numeric values
        data_min = data[qi].min() if hierarchy == "range" else None
        data_max = data[qi].max() if hierarchy == "range" else None
        if hierarchy == "range" and not pd.api.types.is_numeric_dtype(data[qi]):
            print(f"{Fore.RED}Skipping column '{qi}': Contains non-numeric values incompatible with 'range' hierarchy.{Style.RESET_ALL}")
            continue

        try:
            generalized_data[qi] = generalized_data[qi].apply(
                lambda x: generalize_value(x, level, hierarchy, data_min, data_max)
            )
        except Exception as e:
            print(f"{Fore.RED}Error generalizing column '{qi}': {e}{Style.RESET_ALL}")

    return generalized_data


# Generate Generalization Lattice
def generate_lattice(quasi_identifiers, max_levels):
    """Generate the generalization lattice for quasi-identifiers."""
    return list(product(*[range(1, max_level + 1) for max_level in max_levels]))

# ℓ-Diversity: Entropy-Based Check (Iterative)
def check_entropy_l_diversity(data, quasi_identifiers, sensitive_attr, l, tol=1e-3):
    """
    Check ℓ-diversity based on entropy for a dataset.

    Args:
        data: DataFrame to evaluate.
        quasi_identifiers: List of quasi-identifiers to group the data.
        sensitive_attr: Sensitive attribute for ℓ-diversity evaluation.
        l: ℓ-diversity threshold.
        tol: Tolerance for numerical precision.

    Returns:
        bool: True if ℓ-diversity is satisfied for all groups; otherwise, False.
    """
    # Ensure quasi_identifiers is a list, not a tuple
    if isinstance(quasi_identifiers, tuple):
        quasi_identifiers = list(quasi_identifiers)

    # Ensure all quasi-identifiers exist in the DataFrame
    missing_columns = [qi for qi in quasi_identifiers if qi not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the data: {missing_columns}")

    # Handle null-value columns and warn if any column is entirely null
    for qi in quasi_identifiers:
        if data[qi].isnull().all():
            print(f"Warning: Column '{qi}' contains only null values. Skipping ℓ-diversity checks for this column.")
            return False

    # Group by quasi-identifiers
    try:
        grouped = data.groupby(quasi_identifiers)
    except KeyError as e:
        print("Debug: KeyError during grouping.")
        raise KeyError(f"Grouping failed. Check quasi-identifiers and DataFrame columns: {e}")

    # Check ℓ-diversity in each group
    all_satisfied = True
    divider = "-" * 80
    for group_name, group_data in grouped:
        sensitive_counts = group_data[sensitive_attr].value_counts(normalize=True)
        entropy = -sum(p * np.log(p) for p in sensitive_counts if p > 0)
        log_l = np.log(l)

        # Debugging: Print group details
        print(divider)
        print(Fore.LIGHTBLUE_EX + f"Group: {group_name}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Sensitive Counts: {sensitive_counts.to_dict()}" + Style.RESET_ALL)
        print(Fore.BLUE + f"  Entropy: {entropy:.5f}" + Style.RESET_ALL)
        print(Fore.BLUE + f"  Log(l): {log_l:.5f}" + Style.RESET_ALL)

        # Truncate to two decimal places
        entropy_rounded = round(entropy, 2)
        log_l_rounded = round(log_l, 2)

         # Compare rounded values
        if entropy_rounded < log_l_rounded:
            all_satisfied = False
            print(f"{Fore.RED}ℓ-Diversity Failure: Entropy ({entropy_rounded:.2f}) < Log(l) ({log_l_rounded:.2f}){Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}ℓ-Diversity Satisfied: Entropy ({entropy_rounded:.2f}) >= Log(l) ({log_l_rounded:.2f}){Style.RESET_ALL}")

    return all_satisfied


def check_recursive_l_diversity(data, quasi_identifiers, sensitive_attr, l, c):
    """
    Check if the dataset satisfies recursive ℓ-diversity.

    Args:
        data: The generalized dataset (Pandas DataFrame).
        quasi_identifiers: List of quasi-identifiers (columns) to group the data.
        sensitive_attr: The sensitive attribute column to evaluate.
        l: The ℓ-diversity parameter (minimum distinct sensitive values in each group).
        c: The `c` parameter for recursive ℓ-diversity (strengthens the ℓ-diversity condition).

    Returns:
        bool: True if the dataset satisfies recursive ℓ-diversity, False otherwise.
    """
    grouped = data.groupby(quasi_identifiers)  # Group data by quasi-identifiers
    all_satisfied = True  # Track if all groups satisfy recursive ℓ-diversity

    for group_name, group_data in grouped:
        sensitive_counts = group_data[sensitive_attr].value_counts()  # Count sensitive attribute values
        total_count = sensitive_counts.sum()

        # Top `l` most frequent sensitive values
        top_l_values = sensitive_counts.nlargest(l)
        top_l_sum = top_l_values.sum()

        # Recursive ℓ-diversity condition
        if total_count - top_l_sum < c * top_l_values.iloc[-1]:  # The least frequent value in the top `l`
            print(
                f"{Fore.RED}Group {group_name} fails recursive ℓ-diversity: Total count - Top-{l} sum < c * least frequent value.{Style.RESET_ALL}"
            )
            print(f"  Sensitive Counts: {sensitive_counts.to_dict()}")
            print(f"  Top-{l} Values: {top_l_values.to_dict()}")
            print(f"  Condition: {total_count} - {top_l_sum} < {c} * {top_l_values.iloc[-1]}")
            all_satisfied = False
        else:
            print(
                f"{Fore.GREEN}Group {group_name} satisfies recursive ℓ-diversity: Total count - Top-{l} sum >= c * least frequent value.{Style.RESET_ALL}"
            )
            print(f"  Sensitive Counts: {sensitive_counts.to_dict()}")
            print(f"  Top-{l} Values: {top_l_values.to_dict()}")

    return all_satisfied

def check_k_anonymity(data, quasi_identifiers, k):
    """
    Validate if a dataset satisfies k-anonymity.

    Args:
        data: The generalized DataFrame.
        quasi_identifiers: List of quasi-identifiers (or composite QIs).
        k: The k-anonymity threshold.

    Returns:
        True if k-anonymity is satisfied; False otherwise.
    """
    # Ensure quasi_identifiers is a list, not a tuple
    if isinstance(quasi_identifiers, tuple):
        quasi_identifiers = list(quasi_identifiers)

    # Ensure all quasi-identifiers exist in the DataFrame
    missing_columns = [qi for qi in quasi_identifiers if qi not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the data: {missing_columns}")

    # Ensure there are no empty quasi-identifiers
    for qi in quasi_identifiers:
        if data[qi].isnull().all():
            raise ValueError(f"Column '{qi}' contains only null values, which cannot be used for k-anonymity checks.")

    # Debugging: Check for duplicated columns
    duplicated_columns = data.columns.duplicated()
    if any(duplicated_columns):
        print("Debug: Duplicated Columns", data.columns[duplicated_columns])
    
    # Group by quasi-identifiers
    try:
        grouped = data.groupby(quasi_identifiers)
    except KeyError as e:
        print("Debug: KeyError during grouping.")
        raise KeyError(f"Grouping failed. Check quasi-identifiers and DataFrame columns: {e}")

    # Check group sizes for k-anonymity
    for idx, group in enumerate(grouped):
        if len(group) < k:
            print(f"Debug: Group {idx} size is less than {k}.")
            return False

    print(Fore.GREEN + "Debug: k-Anonymity check passed." + Style.RESET_ALL)
    return True

def redistribute_sensitive_data(data, quasi_identifiers, sensitive_attr, l):
    """
    Redistribute records to ensure ℓ-diversity within generalized groups.

    Args:
        data: Input DataFrame.
        quasi_identifiers: List of quasi-identifiers to group data.
        sensitive_attr: The sensitive attribute for ℓ-diversity.
        l: ℓ-diversity threshold.

    Returns:
        DataFrame: The dataset with redistributed records to meet ℓ-diversity.
    """
    redistributed_data = pd.DataFrame(columns=data.columns)  # Initialize the redistributed dataset
    groups = data.groupby(quasi_identifiers)
    failed_groups = 0  # Track the number of failing groups
    divider = Fore.LIGHTBLACK_EX + "-" * 80 + Style.RESET_ALL

    for group_name, group in groups:
        sensitive_counts = group[sensitive_attr].value_counts(normalize=True)
        print(divider)
        print(Fore.CYAN + f"Group: {group_name}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Sensitive Counts: {sensitive_counts.to_dict()}" + Style.RESET_ALL)

        if len(sensitive_counts) < l:  # Check if group fails ℓ-diversity
            failed_groups += 1
            print(
                Fore.RED
                + f"  Group {group_name} fails ℓ-diversity: Requires {l} distinct sensitive values, but only has {len(sensitive_counts)}."
                + Style.RESET_ALL
            )

            # Redistribute records from other groups
            needed = l - len(sensitive_counts)
            print(Fore.MAGENTA + f"  Redistributing: Need {needed} additional sensitive values." + Style.RESET_ALL)

            for other_group_name, other_group in groups:
                if group_name == other_group_name:
                    continue  # Skip the same group

                candidates = other_group[sensitive_attr].value_counts()
                for sensitive_value, count in candidates.items():
                    if sensitive_value not in sensitive_counts and needed > 0:
                        rows_to_move = other_group[other_group[sensitive_attr] == sensitive_value].head(needed)
                        redistributed_data = pd.concat([redistributed_data, rows_to_move])
                        needed -= len(rows_to_move)
                        print(
                            Fore.GREEN
                            + f"  Moved {len(rows_to_move)} records with sensitive value '{sensitive_value}' from group {other_group_name}."
                            + Style.RESET_ALL
                        )
                    
                    if needed <= 0:
                        break

                if needed <= 0:
                    break

            if needed > 0:
                print(Fore.RED + "  Redistribution failed: Insufficient records to meet ℓ-diversity for this group." + Style.RESET_ALL)
        else:
            print(Fore.GREEN + f"  Group {group_name} satisfies ℓ-diversity." + Style.RESET_ALL)
            redistributed_data = pd.concat([redistributed_data, group])

    print(divider)
    print(
        Fore.BLUE
        + f"ℓ-Diversity Redistribution Complete: {failed_groups} groups required redistribution."
        + Style.RESET_ALL
    )
    return redistributed_data

# **New Feature**: NPD Recursive ℓ-Diversity
def check_npd_recursive_l_diversity(data, quasi_identifiers, sensitive_attrs, l, c):
    """Check Non-Parametric Distributions (NPD) Recursive ℓ-Diversity."""
    grouped = data.groupby(quasi_identifiers)
    all_satisfied = True

    for _, group_data in grouped:
        for sensitive_attr in sensitive_attrs:
            sensitive_counts = group_data[sensitive_attr].value_counts()
            total_count = sensitive_counts.sum()
            top_l_values = sensitive_counts.nlargest(l)
            if total_count - top_l_values.sum() < c * top_l_values.iloc[-1]:
                all_satisfied = False
    return all_satisfied

# **New Feature**: Lattice Search Strategy
def lattice_search(data, quasi_identifiers, sensitive_attr, max_levels, hierarchies, strategy="bottom-up"):
    """Perform a lattice search to find a suitable generalization."""
    lattice = generate_lattice(quasi_identifiers, max_levels)
    if strategy == "top-down":
        lattice = reversed(lattice)

    for levels in lattice:
        generalized_data = apply_generalization(data, quasi_identifiers, levels, max_levels, hierarchies)
        if check_entropy_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l=2):
            return generalized_data
    return None

# def apply_l_diversity(data, quasi_identifiers, sensitive_attr, l, max_levels, hierarchies, dataset_name="Dataset", method="entropy", c=None, strategy="bottom-up"):
#     """
#     Apply k-anonymity first, then ℓ-diversity. Redistribute records, generalize, and fallback to composite QIs.

#     Args:
#         data (pd.DataFrame): The dataset to anonymize.
#         quasi_identifiers (list): List of quasi-identifier column names.
#         sensitive_attr (str): Sensitive attribute column name.
#         l (int): Minimum diversity requirement.
#         max_levels (list): Maximum generalization levels for each quasi-identifier.
#         hierarchies (list): Generalization hierarchies for each quasi-identifier.
#         dataset_name (str): Name of the dataset being processed.
#         method (str): ℓ-diversity method ('entropy', 'recursive', 'npd_recursive').
#         c (float): Parameter for recursive ℓ-diversity methods.
#         strategy (str): Lattice search strategy ('bottom-up' or 'top-down').

#     Returns:
#         pd.DataFrame: Dataset satisfying both k-anonymity and ℓ-diversity.
#     """
#     divider = Fore.LIGHTBLACK_EX + "-" * 80 + Style.RESET_ALL
#     print(divider)
#     print(f"{Fore.CYAN}Applying ℓ-Diversity to {dataset_name} using {method} method...{Style.RESET_ALL}")
#     print(f"{Fore.BLUE}Quasi-Identifiers: {', '.join(quasi_identifiers)}{Style.RESET_ALL}")
#     print(f"{Fore.GREEN}Max Levels: {max_levels}{Style.RESET_ALL}")
#     if method in ["recursive", "npd_recursive"] and c is not None:
#         print(f"{Fore.BLUE}Using c value: {c}{Style.RESET_ALL}")
#     print(f"{Fore.GREEN}Lattice Search Strategy: {strategy}{Style.RESET_ALL}")
#     print(divider)

#     # Step 1: Apply pure k-Anonymity
#     print(f"{Fore.YELLOW}Checking initial k-Anonymity...{Style.RESET_ALL}")
#     generalized_data = apply_k_anonymity(data, quasi_identifiers, k=l)  # Generalization levels removed

#     if check_k_anonymity(generalized_data, quasi_identifiers, k=l):
#         print(f"{Fore.GREEN}Initial dataset satisfies k-Anonymity.{Style.RESET_ALL}")

#         # Check if ℓ-Diversity is already satisfied
#         if check_entropy_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l):
#             print(f"{Fore.GREEN}ℓ-Diversity satisfied using initial k-Anonymity.{Style.RESET_ALL}")
#             return generalized_data
        
#     else:
#         print(f"{Fore.RED}Initial dataset does not satisfy k-Anonymity. Proceeding with generalization...{Style.RESET_ALL}")

#     # Step 2: Explore Generalization Levels
#     print(f"{Fore.YELLOW}Exploring generalization levels...{Style.RESET_ALL}")
#     lattice = generate_lattice(quasi_identifiers, max_levels)
#     if strategy == "top-down":
#         lattice = list(reversed(lattice))
#     print(f"{Fore.YELLOW}Generated lattice with {len(lattice)} combinations of generalization levels.{Style.RESET_ALL}")
#     print(divider)

#     for idx, levels in enumerate(lattice):
#         print(f"{Fore.BLUE}[{idx + 1}/{len(lattice)}] Trying generalization levels: {levels}{Style.RESET_ALL}")

#         # Problem
#         # Apply generalization
#         generalized_data = apply_generalization(data, quasi_identifiers, levels, max_levels, hierarchies)

#         # Validate k-Anonymity
#         if not check_k_anonymity(generalized_data, quasi_identifiers, k=l):
#             print(f"{Fore.RED}k-Anonymity not satisfied at levels: {levels}. Continuing...{Style.RESET_ALL}")
#             continue
        
#         # Validate ℓ-Diversity
#         if method == "entropy":
#             if check_entropy_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l):
#                 print(f"{Fore.GREEN}ℓ-Diversity satisfied using entropy at levels: {levels}{Style.RESET_ALL}")
#                 return generalized_data
            
#         return
    
#         # elif method == "recursive" and c is not None:
#         #     if check_recursive_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l, c):
#         #         print(f"{Fore.GREEN}ℓ-Diversity satisfied using recursive at levels: {levels}{Style.RESET_ALL}")
#         #         return generalized_data

#         # elif method == "npd_recursive" and c is not None:
#         #     if check_npd_recursive_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l, c):
#         #         print(f"{Fore.GREEN}ℓ-Diversity satisfied using NPD Recursive at levels: {levels}{Style.RESET_ALL}")
#         #         return generalized_data

#         print(f"{Fore.RED}ℓ-Diversity not satisfied at levels: {levels}. Trying next combination...{Style.RESET_ALL}")

#     # Step 3: Redistribution
#     print(f"{Fore.YELLOW}Generalization alone failed to achieve ℓ-Diversity. Proceeding with redistribution...{Style.RESET_ALL}")
#     redistributed_data = redistribute_sensitive_data(generalized_data, quasi_identifiers, sensitive_attr, l)
#     if check_entropy_l_diversity(redistributed_data, quasi_identifiers, sensitive_attr, l):
#         print(f"{Fore.GREEN}ℓ-Diversity satisfied after redistribution.{Style.RESET_ALL}")
#         return redistributed_data
#     print(f"{Fore.RED}Redistribution failed to satisfy ℓ-Diversity. Proceeding with suppression for small groups...{Style.RESET_ALL}")

#     # Step 4: Suppression for Small Groups
#     grouped = generalized_data.groupby(quasi_identifiers)
#     for group_name, group_data in grouped:
#         sensitive_counts = group_data[sensitive_attr].value_counts()
#         if len(sensitive_counts) < l:  # Check if group fails ℓ-diversity
#             print(f"{Fore.RED}Group {group_name} fails ℓ-diversity. Suppressing sensitive attribute.{Style.RESET_ALL}")
#             generalized_data.loc[group_data.index, sensitive_attr] = "Suppressed"

#     # Step 5: Composite Quasi-Identifiers
#     print(f"{Fore.YELLOW}Exploring composite quasi-identifiers...{Style.RESET_ALL}")
#     composite_quasi_identifiers = generate_composite_quasi_identifiers(quasi_identifiers)

#     for composite_qi in composite_quasi_identifiers:
#         print(f"{Fore.CYAN}Trying composite quasi-identifiers: {', '.join(composite_qi)}{Style.RESET_ALL}")

#         generalized_data = apply_generalization(data, composite_qi, [1] * len(composite_qi), max_levels, hierarchies)

#         # Validate k-Anonymity
#         if not check_k_anonymity(generalized_data, composite_qi, k=l):
#             print(f"{Fore.RED}k-Anonymity not satisfied for composite QIs: {composite_qi}. Continuing...{Style.RESET_ALL}")
#             continue

#         # Validate ℓ-Diversity
#         if method == "entropy":
#             if check_entropy_l_diversity(generalized_data, composite_qi, sensitive_attr, l):
#                 print(f"{Fore.GREEN}ℓ-Diversity satisfied using entropy with composite QIs: {composite_qi}{Style.RESET_ALL}")
#                 return generalized_data

#     print(f"{Fore.RED}ℓ-Diversity requirements could not be met. Suppressing sensitive attribute...{Style.RESET_ALL}")
#     data[sensitive_attr] = "Suppressed"
#     return data

def apply_l_diversity(data, quasi_identifiers, sensitive_attr, l, max_levels, hierarchies, dataset_name="Dataset", method="entropy", c=None, strategy="bottom-up"):
    """
    Apply k-anonymity first, then ℓ-diversity. Add recursive ℓ-diversity option.

    Args:
        data (pd.DataFrame): The dataset to anonymize.
        quasi_identifiers (list): List of quasi-identifier column names.
        sensitive_attr (str): Sensitive attribute column name.
        l (int): Minimum diversity requirement.
        max_levels (list): Maximum generalization levels for each quasi-identifier.
        hierarchies (list): Generalization hierarchies for each quasi-identifier.
        dataset_name (str): Name of the dataset being processed.
        method (str): ℓ-diversity method ('entropy', 'recursive', 'npd_recursive').
        c (float): Parameter for recursive ℓ-diversity methods.
        strategy (str): Lattice search strategy ('bottom-up' or 'top-down').

    Returns:
        pd.DataFrame: Dataset satisfying both k-anonymity and ℓ-diversity.
    """
    divider = Fore.LIGHTBLACK_EX + "-" * 80 + Style.RESET_ALL
    print(divider)
    print(f"{Fore.CYAN}Applying ℓ-Diversity to {dataset_name} using {method} method...{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Quasi-Identifiers: {', '.join(quasi_identifiers)}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Max Levels: {max_levels}{Style.RESET_ALL}")
    if method == "recursive" and c is not None:
        print(f"{Fore.BLUE}Using c value: {c}{Style.RESET_ALL}")
    print(divider)

    # Step 1: Apply pure k-Anonymity
    generalized_data = apply_k_anonymity(data, quasi_identifiers, k=l)
    if not check_k_anonymity(generalized_data, quasi_identifiers, k=l):
        print(f"{Fore.RED}Initial dataset does not satisfy k-Anonymity. Proceeding with generalization...{Style.RESET_ALL}")

    # Step 2: Explore Generalization Levels
    lattice = generate_lattice(quasi_identifiers, max_levels)
    if strategy == "top-down":
        lattice = list(reversed(lattice))

    for idx, levels in enumerate(lattice):
        print(f"{Fore.BLUE}[{idx + 1}/{len(lattice)}] Trying generalization levels: {levels}{Style.RESET_ALL}")
        generalized_data = apply_generalization(data, quasi_identifiers, levels, max_levels, hierarchies)

        if check_k_anonymity(generalized_data, quasi_identifiers, k=l):
            if method == "entropy":
                if check_entropy_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l):
                    print(f"{Fore.GREEN}ℓ-Diversity satisfied using entropy at levels: {levels}{Style.RESET_ALL}")
                    return generalized_data

            elif method == "recursive" and c is not None:
                if check_recursive_l_diversity(generalized_data, quasi_identifiers, sensitive_attr, l, c):
                    print(f"{Fore.GREEN}ℓ-Diversity satisfied using recursive at levels: {levels}{Style.RESET_ALL}")
                    return generalized_data

    print(f"{Fore.YELLOW}Generalization failed to achieve ℓ-Diversity. Proceeding with redistribution and suppression...{Style.RESET_ALL}")

    # Step 3: Redistribution
    redistributed_data = redistribute_sensitive_data(generalized_data, quasi_identifiers, sensitive_attr, l)

    # Check if ℓ-diversity is satisfied after redistribution
    if method == "recursive":
        if check_recursive_l_diversity(redistributed_data, quasi_identifiers, sensitive_attr, l, c):
            print(f"{Fore.GREEN}ℓ-Diversity satisfied after redistribution.{Style.RESET_ALL}")
            return redistributed_data

    # Step 4: Suppression for Groups Failing ℓ-Diversity
    # Suppress sensitive attributes for groups that fail ℓ-diversity
    grouped = redistributed_data.groupby(quasi_identifiers)  # Suppression should operate on redistributed data
    suppressed_groups = []  # Track groups where suppression was applied

    for group_name, group_data in grouped:
        sensitive_counts = group_data[sensitive_attr].value_counts()

        # Step 4.1: Check if the group satisfies ℓ-diversity
        if len(sensitive_counts) < l:
            print(f"{Fore.RED}Group {group_name} fails ℓ-diversity. Suppressing sensitive attribute for this group.{Style.RESET_ALL}")
            # Suppress the sensitive attribute for all rows in the group
            redistributed_data.loc[group_data.index, sensitive_attr] = "Suppressed"
            # Step 4.2: Track suppressed group for summary reporting
            suppressed_groups.append(group_name)

    # Step 5: Display Suppressed Groups Summary
    if suppressed_groups:
        print(f"\n{Fore.YELLOW}Suppressed Groups Summary:{Style.RESET_ALL}")
        for group_name in suppressed_groups:
            print(f"{Fore.CYAN}Group: {group_name}{Style.RESET_ALL} - Sensitive attributes suppressed.")
    else:
        print(f"{Fore.GREEN}No groups required suppression. ℓ-Diversity satisfied after suppression.{Style.RESET_ALL}")

    # Step 6: Return final dataset
    return redistributed_data  # Return redistributed and suppressed data

def generate_composite_quasi_identifiers(quasi_identifiers):
    """
    Generate composite quasi-identifiers from the given quasi-identifiers.

    Args:
        quasi_identifiers (list): List of single quasi-identifiers.

    Returns:
        list: A list of tuples, each representing a combination of composite quasi-identifiers.
    """
    from itertools import combinations

    composite_qi_list = []
    for r in range(2, len(quasi_identifiers) + 1):
        composite_qi_list.extend(list(combinations(quasi_identifiers, r)))
    return composite_qi_list

def compare_all_groups(data, quasi_identifiers, sensitive_attr, l, c=None, check_types=["k-anonymity", "entropy", "recursive"]):
    """
    Compare all groups and their records after applying k-Anonymity and ℓ-Diversity (entropy and recursive).
    
    Args:
        data (pd.DataFrame): The original dataset.
        quasi_identifiers (list): List of quasi-identifiers.
        sensitive_attr (list): Sensitive attribute(s).
        l (int): ℓ-diversity threshold.
        c (float): Optional parameter for recursive ℓ-Diversity.
        check_types (list): List of checks to perform ("k-anonymity", "entropy", "recursive").
    """
    print(Fore.BLUE + "Starting comparison of groups across anonymization stages..." + Style.RESET_ALL)
    
    # Apply k-Anonymity
    k_anonymized_data = apply_k_anonymity(data, quasi_identifiers, k=l)
    
    # Apply ℓ-Diversity (Entropy-based) on k-anonymized data
    l_diverse_entropy_data = apply_l_diversity(
        data=k_anonymized_data,
        quasi_identifiers=quasi_identifiers,
        sensitive_attr=sensitive_attr,
        l=l,
        max_levels=[4, 1, 1, 2],  # Example max levels
        hierarchies=["range", "suppression", "taxonomy", "taxonomy"],
        method="entropy"
    )

    # Apply Recursive ℓ-Diversity
    if "recursive" in check_types:
        l_diverse_recursive_data = apply_l_diversity(
            data=k_anonymized_data,
            quasi_identifiers=quasi_identifiers,
            sensitive_attr=sensitive_attr,
            l=l,
            max_levels=[4, 1, 1, 2],  # Example max levels
            hierarchies=["range", "suppression", "taxonomy", "taxonomy"],
            method="recursive",
            c=c
        )
    else:
        l_diverse_recursive_data = None

    # Group-by and display comparison for each step
    for group_name, k_group in k_anonymized_data.groupby(quasi_identifiers):
        print("\n" + "-" * 80)
        print(f"{Fore.CYAN}Group: {group_name} (k-Anonymized){Style.RESET_ALL}")
        print(k_group.to_csv(index=False, header=False).strip())  # Print raw records in the group
        
        # ℓ-Diversity Entropy-based
        if "entropy" in check_types:
            l_entropy_group = l_diverse_entropy_data[l_diverse_entropy_data[quasi_identifiers].apply(tuple, axis=1) == group_name]
            print(f"\n{Fore.YELLOW}Group: {group_name} (ℓ-Diversity - Entropy){Style.RESET_ALL}")
            print(l_entropy_group.to_csv(index=False, header=False).strip())  # Print raw records

        # Recursive ℓ-Diversity
        if l_diverse_recursive_data is not None:
            l_recursive_group = l_diverse_recursive_data[l_diverse_recursive_data[quasi_identifiers].apply(tuple, axis=1) == group_name]
            print(f"\n{Fore.MAGENTA}Group: {group_name} (ℓ-Diversity - Recursive){Style.RESET_ALL}")
            print(l_recursive_group.to_csv(index=False, header=False).strip())  # Print raw records



if __name__ == "__main__":
    # File paths for raw datasets
    adult_data_path = "./data/original_dataset/adult_dataset.csv"
    lands_end_data_path = "./data/original_dataset/lands_end_dataset.csv"

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
        "l": 3,  # ℓ-diversity threshold
    }
    adult_quasi_identifiers = ["Age", "Gender", "Education", "Work Class"]
    # adult_quasi_identifiers = ["Age", "Gender", "Education"]
    # adult_quasi_identifiers = ["Age", "Gender"]
    # adult_sensitive_attr = ["Salary Class"]  # Multi-attribute ℓ-diversity support
    adult_sensitive_attr = ["Occupation"]  # Multi-attribute ℓ-diversity support

    # Parameters for Lands End Dataset
    lands_end_params = {
        "max_levels": [5, 1, 1, 4],  # Generalization levels for quasi-identifiers
        "hierarchies": ["round", "suppression", "suppression", "round"],
        "l": 3,  # ℓ-diversity threshold
    }
    lands_end_quasi_identifiers = ["Zipcode", "Gender", "Style", "Price"]
    # lands_end_quasi_identifiers = ["Zipcode", "Gender", "Price"]
    # lands_end_quasi_identifiers = ["Zipcode", "Gender"]
    lands_end_sensitive_attr = ["Cost"]  # Multi-attribute ℓ-diversity support

    # Step 1: Apply ℓ-Diversity for the Adult Dataset
    print(Fore.BLUE + "Applying k-Anonymity and ℓ-Diversity for Adult Dataset..." + Style.RESET_ALL)
    adult_result = apply_l_diversity(
        data=adult_df,
        quasi_identifiers=adult_quasi_identifiers,
        sensitive_attr=adult_sensitive_attr,
        l=adult_params["l"],
        max_levels=adult_params["max_levels"],
        hierarchies=adult_params["hierarchies"],
        dataset_name="Adult Dataset"
    )

    if adult_result is not None:
        output_path = "./data/l_diversity_dataset/adult_l_diverse.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        adult_result.to_csv(output_path, index=False)
        print(Fore.GREEN + f"ℓ-Diversity applied successfully to the Adult Dataset. Results saved to '{output_path}'." + Style.RESET_ALL)
    else:
        print(Fore.RED + "ℓ-Diversity could not be satisfied for the Adult Dataset." + Style.RESET_ALL)

    # Step 2: Apply NPD Recursive ℓ-Diversity for the Adult Dataset
    print(Fore.BLUE + "Applying NPD Recursive ℓ-Diversity for Adult Dataset..." + Style.RESET_ALL)
    result_npd_adult = check_npd_recursive_l_diversity(adult_df, adult_quasi_identifiers, adult_sensitive_attr, l=2, c=0.5)
    print(f"{Fore.RED}NPD Recursive ℓ-Diversity for Adult Dataset: {'Satisfied' if result_npd_adult else 'Not Satisfied'}{Style.RESET_ALL}")

    # Step 4: Apply ℓ-Diversity for the Lands End Dataset
    print(Fore.BLUE + "Applying k-Anonymity and ℓ-Diversity for Lands End Dataset..." + Style.RESET_ALL)
    lands_end_result = apply_l_diversity(
        data=lands_end_df,
        quasi_identifiers=lands_end_quasi_identifiers,
        sensitive_attr=lands_end_sensitive_attr,
        l=lands_end_params["l"],
        max_levels=lands_end_params["max_levels"],
        hierarchies=lands_end_params["hierarchies"],
        dataset_name="Lands End Dataset"
    )
    if lands_end_result is not None:
        output_path = "./data/l_diversity_dataset/lands_end_l_diverse.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lands_end_result.to_csv(output_path, index=False)
        print(Fore.GREEN + f"ℓ-Diversity applied successfully to the Lands End Dataset. Results saved to '{output_path}'." + Style.RESET_ALL)
    else:
        print(Fore.RED + "ℓ-Diversity could not be satisfied for the Lands End Dataset." + Style.RESET_ALL)

    # Step 5: Apply NPD Recursive ℓ-Diversity for the Lands End Dataset
    print(Fore.BLUE + "Applying NPD Recursive ℓ-Diversity for Lands End Dataset..." + Style.RESET_ALL)
    result_npd_lands_end = check_npd_recursive_l_diversity(lands_end_df, lands_end_quasi_identifiers, lands_end_sensitive_attr, l=2, c=0.5)
    print(Fore.RED + f"NPD Recursive ℓ-Diversity for Lands End Dataset: {'Satisfied' if result_npd_lands_end else 'Not Satisfied'}" + Style.RESET_ALL)  

    # Apply k-Anonymity and ℓ-Diversity
    generalization_levels = {qi: 1 for qi in adult_quasi_identifiers}  # Start with level 1 for all QIs

    # Apply k-Anonymity
    l_diversity_result = apply_l_diversity(
        adult_df,
        adult_quasi_identifiers,
        adult_sensitive_attr,
        l=2,
        max_levels=adult_params["max_levels"],
        hierarchies=adult_params["hierarchies"],
        dataset_name="Adult Dataset"
    )

    # Compare
    # compare_all_groups(
    #     data=adult_df,
    #     quasi_identifiers=adult_quasi_identifiers,
    #     sensitive_attr=adult_sensitive_attr,
    #     l=2,
    #     c=2,  # Only required for recursive ℓ-Diversity
    #     check_types=["k-anonymity", "entropy", "recursive"]
    # )
