import random
import pandas as pd
from colorama import Fore, Style

# Helper to create sensitive values
def create_sensitive_values(categories, num_records, weights=None):
    """Generate sensitive attribute values with balanced distribution."""
    return random.choices(categories, weights=weights, k=num_records)

# Create Adult Dataset
def create_realistic_adult_dataset(num_records=1000):
    """Create a realistic Adult Dataset as per paper specifications."""
    salary_weights = [0.5, 0.5]  # Balanced distribution for Salary Class
    occupation_weights = [1 / 41] * 41  # Uniform distribution for 41 occupations

    data = {
        "Age": [random.randint(18, 90) for _ in range(num_records)],  # Domain size = 74
        "Gender": random.choices(["Male", "Female"], weights=[0.48, 0.52], k=num_records),  # Domain size = 2
        "Race": random.choices(["White", "Black", "Asian", "Hispanic", "Other"], k=num_records),  # Domain size = 5
        "Marital Status": random.choices(
            ["Married", "Single", "Divorced", "Widowed", "Separated", "Partnered", "Other"], k=num_records
        ),  # Domain size = 7
        "Education": random.choices(
            ["High School", "Some College", "Bachelor's", "Master's", "Doctorate", "Associate", "None"],
            k=num_records
        ),  # Domain size = 16
        "Native Country": random.choices(
            [f"Country {i}" for i in range(1, 42)], k=num_records
        ),  # Domain size = 41
        "Work Class": random.choices(
            ["Private", "Self-Employed", "Government", "Unemployed", "Retired", "Student", "Other"], k=num_records
        ),  # Domain size = 7
        "Salary Class": create_sensitive_values(["<=50K", ">50K"], num_records, weights=salary_weights),  # Domain size = 2
        "Occupation": create_sensitive_values(
            [f"Occupation {i}" for i in range(1, 42)], num_records, weights=occupation_weights
        ),  # Domain size = 41
    }

    # Save to CSV
    df = pd.DataFrame(data)
    output_path = "./data/original_dataset/adult_dataset.csv"
    df.to_csv(output_path, index=False)
    print(Fore.GREEN + f"Adult dataset created and saved as '{output_path}'." + Style.RESET_ALL)

# Create Lands End Dataset
def create_realistic_lands_end_dataset(num_records=1000):
    """Create a realistic Lands End Dataset as per paper specifications."""
    cost_weights = [1 / 147] * 147  # Uniform distribution for 147 cost values
    price_weights = [1 / 346] * 346  # Uniform distribution for 346 price values

    data = {
        "Zipcode": [random.randint(10000, 99999) for _ in range(num_records)],  # Domain size = 31953
        "Order Date": [
            f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(num_records)
        ],  # Domain size = 320
        "Gender": random.choices(["Male", "Female"], weights=[0.48, 0.52], k=num_records),  # Domain size = 2
        "Style": random.choices(
            [f"Style {i}" for i in range(1, 1510)], k=num_records
        ),  # Domain size = 1509
        "Price": create_sensitive_values(
            [i * 10 for i in range(1, 347)], num_records, weights=price_weights
        ),  # Domain size = 346
        "Quantity": random.choices([random.randint(1, 10) for _ in range(10)], k=num_records),  # Random quantities (1-10)
        "Shipment": random.choices(["Air", "Sea", "Rail"], weights=[0.5, 0.4, 0.1], k=num_records),  # Domain size = 2
        "Cost": create_sensitive_values(
            [i * 50 for i in range(2, 149)], num_records, weights=cost_weights
        ),  # Domain size = 147
    }

    # Save to CSV
    df = pd.DataFrame(data)
    output_path = "./data/original_dataset/lands_end_dataset.csv"
    df.to_csv(output_path, index=False)
    print(Fore.GREEN + f"Lands End dataset created and saved as '{output_path}'." + Style.RESET_ALL)

# Generate datasets
create_realistic_adult_dataset()
create_realistic_lands_end_dataset()
