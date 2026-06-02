import os
import re
import math
import pandas as pd

from collections import Counter

# =========================================================
# FEATURE ALIASES
# =========================================================

FEATURE_ALIASES = {

    # =====================================================
    # Episodes / Emergency
    # =====================================================

    "episodes_wheezing_year": [
        # "episodes",
        "asthma episodes",
        "asthma attacks",
        "frequent attacks",
        "respiratory episodes",
        "breathing episodes"
    ],

    "emergency_asthma_visits": [
        "emergency visit",
        "hospital visit",
        "emergency care",
        "er visit",
        "urgent care",
        "hospitalization"
    ],

    # =====================================================
    # Sleep Disorder
    # =====================================================

    "sleep_disorder_2.0": [
        "sleep disorder",
        "poor sleep",
        "sleep problems",
        "disturbed sleep",
        "sleep difficulty",
        "irregular sleep",
        "sleep issues",
        "trouble sleeping"
    ],

    # =====================================================
    # Eosinophils
    # =====================================================

    "eosinophil_percent": [
        # "eosinophils",
        "high eosinophils",
        "elevated eosinophils",
        "immune inflammation",
        "allergic inflammation",
        "airway inflammation"
    ],

    # =====================================================
    # Age
    # =====================================================

    "age_years": [
        # "age",
        "older age",
        "elderly",
        "advanced age",
        "aging",
        "older adult",
        "increasing age"
    ],

    # =====================================================
    # Chronic Bronchitis
    # =====================================================

    "chronic_bronchitis_2.0": [
        "chronic bronchitis",
        "bronchitis",
        "lung disease",
        "chronic respiratory disease",
        "airway disease",
        "breathing illness"
    ],

    # =====================================================
    # Income / Poverty
    # =====================================================

    "income_poverty_ratio": [
        "low income",
        "income level",
        "economic condition",
        "financial status",
        "poverty",
        "socioeconomic condition",
        "limited resources"
    ],

    # =====================================================
    # Weight
    # =====================================================

    "weight_kg": [
        "weight",
        "body weight",
        "high weight",
        "increased weight",
        "heavy body weight"
    ],

    # =====================================================
    # BMI
    # =====================================================

    "bmi": [
        "body mass index",
        "obesity",
        "obese",
        "overweight",
        "high bmi",
        "increased bmi",
        "excess body fat"
    ],

    # =====================================================
    # Waist Circumference
    # =====================================================

    "waist_circumference_cm": [
        "waist circumference",
        "abdominal obesity",
        "central obesity",
        "belly fat",
        "large waist",
        "increased waist size"
    ],

    # =====================================================
    # Woke Chest
    # =====================================================

    "woke_breathlessness": [
        "woke with chest discomfort",
        "night chest tightness",
        "chest tightness",
        "night breathing difficulty",
        "wheezing at night",
        "night respiratory symptoms",
        "chest symptoms during sleep"
    ]
}


# =========================================================
# LOAD EXPLANATIONS
# =========================================================

def load_explanations(folder_path):

    explanations = []

    for file in sorted(os.listdir(folder_path)):

        if file.endswith(".txt"):

            file_path = os.path.join(folder_path, file)

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                explanations.append(f.read())

    return explanations

# =========================================================
# EXACT FIDELITY
# =========================================================

def compute_exact_fidelity(
    explanation,
    top_features,
    aliases
):

    explanation = explanation.lower()

    mentioned = 0

    for _, row in top_features.iterrows():

        feature = row["Feature"]

        names = [feature.lower()] + aliases.get(feature, [])

        if any(re.search(rf"\b{re.escape(name.lower())}\b", explanation)
            for name in names
        ):

            mentioned += 1

    return mentioned / len(top_features)

# =========================================================
# WEIGHTED FIDELITY
# =========================================================

def compute_weighted_fidelity(
    explanation,
    top_features,
    aliases
):

    explanation = explanation.lower()

    matched_weight = 0
    total_weight = 0

    for _, row in top_features.iterrows():

        feature = row["Feature"]

        importance = abs(row["Mean_SHAP_Value"])

        total_weight += importance

        names = [feature.lower()] + aliases.get(feature, [])

        if any(name.lower() in explanation for name in names):

            matched_weight += importance

    return matched_weight / total_weight

# =========================================================
# EXTRACT MENTIONED FEATURES
# =========================================================

def extract_mentioned_features(
    explanation,
    top_features,
    aliases
):

    explanation = explanation.lower()

    mentioned = set()

    for _, row in top_features.iterrows():

        feature = row["Feature"]

        names = [feature.lower()] + aliases.get(feature, [])

        if any(name.lower() in explanation for name in names):

            mentioned.add(feature)

    return mentioned

# =========================================================
# ENTROPY
# =========================================================

def compute_entropy(feature_sets):

    total_runs = len(feature_sets)

    counter = Counter()

    for fs in feature_sets:

        for f in set(fs):

            counter[f] += 1

    entropy = 0

    for count in counter.values():

        p = count / total_runs

        entropy -= p * math.log2(p)

    max_entropy = (
        math.log2(len(counter))
        if len(counter) > 1
        else 1
    )

    normalized_entropy = entropy / max_entropy

    return normalized_entropy

# =========================================================
# EXACT MATCH ACCURACY
# =========================================================

def compute_exact_match_accuracy(
    feature_sets,
    top_features
):

    target_set = set(top_features["Feature"])

    exact_matches = 0

    for fs in feature_sets:

        if fs == target_set:

            exact_matches += 1

    return exact_matches / len(feature_sets)

# =========================================================
# RUN FEEM
# =========================================================

def run_feem_evaluation(

    model_name,
    explanation_folder,
    shap_csv_path

):

    # -----------------------------------------------------

    top_features = pd.read_csv(
        shap_csv_path
    ).head(10)

    explanations = load_explanations(
        explanation_folder
    )

    # -----------------------------------------------------

    fidelity_scores = []

    weighted_scores = []

    feature_sets = []

    # -----------------------------------------------------

    for exp in explanations:

        fidelity = compute_exact_fidelity(
            exp,
            top_features,
            FEATURE_ALIASES
        )

        weighted = compute_weighted_fidelity(
            exp,
            top_features,
            FEATURE_ALIASES
        )

        features = extract_mentioned_features(
            exp,
            top_features,
            FEATURE_ALIASES
        )

        fidelity_scores.append(fidelity)

        weighted_scores.append(weighted)

        feature_sets.append(features)

    # -----------------------------------------------------

    avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)

    avg_weighted = sum(weighted_scores) / len(weighted_scores)

    entropy = compute_entropy(feature_sets)

    exact_match = compute_exact_match_accuracy(
        feature_sets,
        top_features
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    os.makedirs(
        "outputs/feem_results",
        exist_ok=True
    )

    results_df = pd.DataFrame({

        "Run": list(range(1, len(explanations)+1)),

        "Fidelity": fidelity_scores,

        "Weighted_Fidelity": weighted_scores
    })

    results_path = (
        f"outputs/feem_results/"
        f"{model_name}_feem_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # -----------------------------------------------------

    summary = {

        "Model": model_name,

        "Average_Fidelity": round(avg_fidelity, 3),

        "Average_Weighted_Fidelity": round(avg_weighted, 3),

        "Entropy": round(entropy, 3),

        "Exact_Match_Accuracy": round(exact_match, 3)
    }

    summary_df = pd.DataFrame([summary])

    summary_path = (
        f"outputs/feem_results/"
        f"{model_name}_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # -----------------------------------------------------

    print("\n=================================")
    print(f"FEEM RESULTS — {model_name}")
    print("=================================")

    print(f"Average Fidelity: {avg_fidelity:.3f}")

    print(f"Weighted Fidelity: {avg_weighted:.3f}")

    print(f"Entropy: {entropy:.3f}")

    print(f"Exact Match Accuracy: {exact_match:.3f}")

    print("=================================\n")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    SHAP_PATH = (
        "outputs/shap/"
        "shap_feature_importance.csv"
    )

    # -----------------------------------------------------
    # LLAMA
    # -----------------------------------------------------

    run_feem_evaluation(

        model_name="llama",

        explanation_folder=
        "outputs/llm_outputs/llama",

        shap_csv_path=SHAP_PATH
    )

    # -----------------------------------------------------
    # MISTRAL
    # -----------------------------------------------------

    run_feem_evaluation(

        model_name="mistral",

        explanation_folder=
        "outputs/llm_outputs/mistral",

        shap_csv_path=SHAP_PATH
    )

