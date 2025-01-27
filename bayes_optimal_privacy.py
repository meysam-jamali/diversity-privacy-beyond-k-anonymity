import pandas as pd
from k_anonymity import apply_k_anonymity

def compute_priors(data, sensitive_attr):
    """
    Compute prior probabilities of the sensitive attribute.
    """
    priors = data[sensitive_attr].value_counts(normalize=True).to_dict()
    return priors

def compute_posteriors(anonymized_data, sensitive_attr, quasi_identifiers):
    """
    Compute posterior probabilities of the sensitive attribute given the anonymized data.
    """
    group_posteriors = {}
    grouped = anonymized_data.groupby(quasi_identifiers)
    
    for group, indices in grouped.groups.items():
        group_data = anonymized_data.loc[indices]
        posterior = group_data[sensitive_attr].value_counts(normalize=True).to_dict()
        group_posteriors[group] = posterior
    
    return group_posteriors

def measure_privacy_loss(priors, posteriors):
    """
    Calculate the privacy loss for each group based on priors and posteriors.
    """
    privacy_loss = {}
    
    for group, posterior in posteriors.items():
        group_loss = 0
        for value, posterior_prob in posterior.items():
            prior_prob = priors.get(value, 0)
            group_loss += abs(posterior_prob - prior_prob)
        privacy_loss[group] = group_loss
    
    return privacy_loss

def bayes_optimal_privacy(data, anonymized_data, sensitive_attr, quasi_identifiers):
    """
    Compute priors, posteriors, and privacy loss for Bayes-optimal privacy evaluation.
    """
    priors = compute_priors(data, sensitive_attr)
    posteriors = compute_posteriors(anonymized_data, sensitive_attr, quasi_identifiers)
    privacy_loss = measure_privacy_loss(priors, posteriors)
    return priors, posteriors, privacy_loss

if __name__ == "__main__":
    # Quasi-identifiers for each database
    quasi_identifiers_adult = ["Age", "Gender", "Race", "Marital Status"]
    quasi_identifiers_lands_end = ["Zipcode", "Gender"]

    # Sensitive attributes
    sensitive_attr_adult = "Salary Class"
    sensitive_attr_lands_end = "Style"

    # Load Original and Anonymized Data for Adult Database
    original_adult = pd.read_csv("./data/original_dataset/adult_dataset.csv")
    anonymized_adult = apply_k_anonymity(original_adult, quasi_identifiers_adult, k=4)

    # Bayes-Optimal Privacy Evaluation for Adult Database
    print("\nEvaluating Bayes-Optimal Privacy for Adult Database:")
    priors_adult, posteriors_adult, privacy_loss_adult = bayes_optimal_privacy(
        original_adult, anonymized_adult, sensitive_attr_adult, quasi_identifiers_adult
    )

    # Display Results for Adult Database
    print("\nPriors (Adult Database):")
    print(priors_adult)
    print("\nPosteriors (Adult Database):")
    for group, posterior in posteriors_adult.items():
        print(f"Group {group}: {posterior}")
    print("\nPrivacy Loss (Adult Database):")
    for group, loss in privacy_loss_adult.items():
        print(f"Group {group}: {loss}")

    # Load Original and Anonymized Data for Lands End Database
    original_lands = pd.read_csv("./data/original_dataset/lands_end_dataset.csv")
    anonymized_lands = apply_k_anonymity(original_lands, quasi_identifiers_lands_end, k=4)

    # Bayes-Optimal Privacy Evaluation for Lands End Database
    print("\nEvaluating Bayes-Optimal Privacy for Lands End Database:")
    priors_lands, posteriors_lands, privacy_loss_lands = bayes_optimal_privacy(
        original_lands, anonymized_lands, sensitive_attr_lands_end, quasi_identifiers_lands_end
    )

    # Display Results for Lands End Database
    print("\nPriors (Lands End Database):")
    print(priors_lands)
    print("\nPosteriors (Lands End Database):")
    for group, posterior in posteriors_lands.items():
        print(f"Group {group}: {posterior}")
    print("\nPrivacy Loss (Lands End Database):")
    for group, loss in privacy_loss_lands.items():
        print(f"Group {group}: {loss}")
