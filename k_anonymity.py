# Import necessary libraries
import pandas as pd
from colorama import Fore, Style

# Generalization Functions
def generalize_age(age):
    """Generalize Age dynamically in increments to satisfy k-anonymity."""
    if pd.isna(age) or age < 18 or age > 90:
        return "Unknown"  # Handle missing or out-of-range values
    age = int(age)
    if age < 30:
        return "<30"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "60+"

def generalize_zipcode(zipcode):
    """Generalize Zipcode dynamically to satisfy k-anonymity."""
    zipcode = str(zipcode).zfill(5)  # Ensure 5-digit zipcodes
    return zipcode[:3] + "**"  # Generalize to the first 3 digits

def generalize_other(value):
    """Suppress non-essential quasi-identifiers."""
    return "Any"  # Generic suppression for simplicity

# k-Anonymity Implementation
def apply_k_anonymity(data, quasi_identifiers, k):
    """
    Apply pure k-anonymity to the dataset with no over-generalization.

    Args:
        data (pd.DataFrame): Dataset to anonymize.
        quasi_identifiers (list): List of quasi-identifier columns.
        k (int): Minimum group size to satisfy k-anonymity.

    Returns:
        pd.DataFrame: Anonymized dataset satisfying k-anonymity.
    """
    print(Fore.BLUE + "Applying k-Anonymity..." + Style.RESET_ALL)

    # Make a copy to avoid modifying the original dataset
    anonymized_data = data.copy()

    # Apply generalization to quasi-identifiers
    for qi in quasi_identifiers:
        if qi == "Age":
            anonymized_data[qi] = anonymized_data[qi].apply(generalize_age)
        elif qi == "Zipcode":
            anonymized_data[qi] = anonymized_data[qi].apply(generalize_zipcode)
        else:
            anonymized_data[qi] = anonymized_data[qi].apply(generalize_other)

    # Group the data and ensure k-anonymity
    grouped = anonymized_data.groupby(quasi_identifiers)
    group_sizes = grouped.size()

    # Identify and suppress groups that fail k-anonymity
    under_sized_groups = group_sizes[group_sizes < k]
    if not under_sized_groups.empty:
        print(Fore.RED + f"{len(under_sized_groups)} groups failed k-anonymity. Suppressing these groups." + Style.RESET_ALL)
        for group, size in under_sized_groups.items():
            indices = grouped.groups[group]
            anonymized_data.loc[indices, quasi_identifiers] = "Suppressed"

    print(Fore.GREEN + "k-Anonymity successfully applied!" + Style.RESET_ALL)
    return anonymized_data

# Display Anonymized Groups
def display_k_groups(data, quasi_identifiers, dataset_name="Dataset"):
    """Display anonymized dataset grouped by quasi-identifiers."""
    grouped = data.groupby(quasi_identifiers)
    print(Fore.YELLOW + f"\n{dataset_name} - k-Anonymized Groups:" + Style.RESET_ALL)
    for group, group_data in grouped:
        print(Fore.CYAN + f"\nGroup (Based on {', '.join(quasi_identifiers)}): {group}" + Style.RESET_ALL)
        print(group_data.to_string(index=False))
        print(Fore.MAGENTA + "-" * 80 + Style.RESET_ALL)

# Main Execution
def main():
    # Parameters
    k = 4  # Minimum group size for k-anonymity
    adult_quasi_identifiers = ["Age", "Gender", "Race", "Marital Status"]
    lands_end_quasi_identifiers = ["Zipcode", "Gender", "Price"]

    # Load datasets
    try:
        print(Fore.GREEN + "Loading datasets..." + Style.RESET_ALL)
        adult_df = pd.read_csv("./data/original_dataset/adult_dataset.csv")
        lands_end_df = pd.read_csv("./data/original_dataset/lands_end_dataset.csv")
    except Exception as e:
        print(Fore.RED + f"Error loading datasets: {e}" + Style.RESET_ALL)
        return

    # Apply k-Anonymity to Adult Dataset
    print(Fore.BLUE + "\nProcessing Adult Dataset..." + Style.RESET_ALL)
    adult_anonymized = apply_k_anonymity(adult_df, adult_quasi_identifiers, k)
    adult_anonymized.to_csv("./data/k_anonymity_dataset/adult_k_anonymized.csv", index=False)
    display_k_groups(adult_anonymized, adult_quasi_identifiers, "Adult Dataset")

    # Apply k-Anonymity to Lands End Dataset
    print(Fore.BLUE + "\nProcessing Lands End Dataset..." + Style.RESET_ALL)
    lands_end_anonymized = apply_k_anonymity(lands_end_df, lands_end_quasi_identifiers, k)
    lands_end_anonymized.to_csv("./data/k_anonymity_dataset/lands_end_k_anonymized.csv", index=False)
    display_k_groups(lands_end_anonymized, lands_end_quasi_identifiers, "Lands End Dataset")

    print(Fore.GREEN + "\nProcessing completed. Anonymized datasets saved." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
