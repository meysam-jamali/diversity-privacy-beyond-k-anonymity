import pandas as pd
from colorama import Fore, Style

# Generalization Functions
def generalize_age(age, level=1):
    if pd.isna(age) or age < 18 or age > 90:
        return "Unknown"
    age = int(age)
    if level == 1:
        step = 5
    elif level == 2:
        step = 10
    elif level == 3:
        step = 20
    elif level == 4:
        return "*"
    else:
        return "Unknown"
    lower_bound = (age // step) * step
    upper_bound = lower_bound + step - 1
    return f"{lower_bound}-{upper_bound}"

def generalize_gender(gender, level=1):
    if level == 1:
        return gender
    elif level == 2:
        return "*"
    return "Unknown"

def generalize_race(race, level=1):
    if level == 1:
        return race
    elif level == 2:
        return "*"
    return "Unknown"

def generalize_marital_status(status, level=1):
    if level == 1:
        return status
    elif level == 2:
        if status in ["Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"]:
            return "Married"
        else:
            return "Single"
    return "Unknown"

# k-Anonymity Implementation
def apply_k_anonymity(data, quasi_identifiers, k, generalization_levels=None):
    if generalization_levels is None:
        generalization_levels = {qi: 1 for qi in quasi_identifiers}

    print(Fore.BLUE + "Applying k-Anonymity..." + Style.RESET_ALL)

    anonymized_data = data.copy()

    for qi in quasi_identifiers:
        level = generalization_levels.get(qi, 1)
        if qi == "Age":
            anonymized_data[qi] = anonymized_data[qi].apply(lambda x: generalize_age(x, level))
        if qi == "Gender":
            anonymized_data[qi] = anonymized_data[qi].apply(lambda x: generalize_gender(x, level))
        if qi == "Race":
            anonymized_data[qi] = anonymized_data[qi].apply(lambda x: generalize_race(x, level))
        if qi == "Marital Status":
            anonymized_data[qi] = anonymized_data[qi].apply(lambda x: generalize_marital_status(x, level))

    grouped = anonymized_data.groupby(quasi_identifiers)
    group_sizes = grouped.size()
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
    grouped = data.groupby(quasi_identifiers)
    print(Fore.YELLOW + f"\n{dataset_name} - k-Anonymized Groups:" + Style.RESET_ALL)
    for group, group_data in grouped:
        print(Fore.CYAN + f"\nGroup (Based on {', '.join(quasi_identifiers)}): {group}" + Style.RESET_ALL)
        print(group_data.to_string(index=False))
        print(Fore.MAGENTA + "-" * 80 + Style.RESET_ALL)

# Main Execution
def main():
    k = 4
    adult_quasi_identifiers = ["Age", "Gender", "Race", "Marital Status"]
    lands_end_quasi_identifiers = ["Zipcode", "Gender", "Price"]

    try:
        print(Fore.GREEN + "Loading datasets..." + Style.RESET_ALL)
        adult_df = pd.read_csv("./data/original_dataset/adult_dataset.csv")
        lands_end_df = pd.read_csv("./data/original_dataset/lands_end_dataset.csv")
    except Exception as e:
        print(Fore.RED + f"Error loading datasets: {e}" + Style.RESET_ALL)
        return

    print(Fore.BLUE + "\nProcessing Adult Dataset..." + Style.RESET_ALL)
    generalization_levels_adult = {"Age": 2, "Gender": 1, "Race": 2, "Marital Status": 2}
    adult_anonymized = apply_k_anonymity(adult_df, adult_quasi_identifiers, k, generalization_levels_adult)
    adult_anonymized.to_csv("./data/k_anonymity_dataset/adult_k_anonymized.csv", index=False)
    display_k_groups(adult_anonymized, adult_quasi_identifiers, "Adult Dataset")

    print(Fore.BLUE + "\nProcessing Lands End Dataset..." + Style.RESET_ALL)
    generalization_levels_lands_end = {"Zipcode": 1, "Gender": 1, "Price": 1}
    lands_end_anonymized = apply_k_anonymity(lands_end_df, lands_end_quasi_identifiers, k, generalization_levels_lands_end)
    lands_end_anonymized.to_csv("./data/k_anonymity_dataset/lands_end_k_anonymized.csv", index=False)
    display_k_groups(lands_end_anonymized, lands_end_quasi_identifiers, "Lands End Dataset")

    print(Fore.GREEN + "\nProcessing completed. Anonymized datasets saved." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
