import os
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sentence_transformers import (
    SentenceTransformer,
    util
)

FEATURE_DESCRIPTIONS = {

    "episodes_wheezing_year":
        "Frequent asthma and wheezing episodes",

    "emergency_asthma_visits":
        "Emergency hospital visits due to asthma",

    "sleep_disorder_2.0":
        "Sleep disorder and poor sleep quality",

    "eosinophil_percent":
        "Elevated eosinophil levels and airway inflammation",

    "age_years":
        "Older age and aging-related respiratory risk",

    "chronic_bronchitis_2.0":
        "Chronic bronchitis and chronic respiratory disease",

    "income_poverty_ratio":
        "Low socioeconomic and income status",

    "weight_kg":
        "High body weight and obesity",

    "bmi":
        "Body mass index and obesity",

    "waist_circumference_cm":
        "Abdominal obesity and waist circumference",

    "woke_breathlessness":
        "Nighttime breathlessness and chest tightness"
}



# =========================================================
# CONFIG
# =========================================================

SIMILARITY_THRESHOLD = 0.45

# =========================================================
# LOAD MODEL
# =========================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# LOAD EXPLANATIONS
# =========================================================

def load_explanations(folder_path):

    explanations = []

    for file in sorted(os.listdir(folder_path)):

        if file.endswith(".txt"):

            file_path = os.path.join(
                folder_path,
                file
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                explanations.append(f.read())

    return explanations

# =========================================================
# SPLIT SENTENCES
# =========================================================

def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]

# =========================================================
# SEMANTIC FIDELITY
# =========================================================

def compute_semantic_fidelity(
    explanation,
    top_features
):

    sentences = split_sentences(
        explanation
    )

    matched = 0

    similarity_scores = []

    # -----------------------------------------------------

    for _, row in top_features.iterrows():

        feature = FEATURE_DESCRIPTIONS.get(
            row["Feature"],
            row["Feature"]
        )

        feature_embedding = model.encode(
            feature,
            convert_to_tensor=True
        )

        max_similarity = 0

        # -------------------------------------------------

        for sent in sentences:

            sent_embedding = model.encode(
                sent,
                convert_to_tensor=True
            )

            similarity = util.cos_sim(
                feature_embedding,
                sent_embedding
            ).item()

            max_similarity = max(
                max_similarity,
                similarity
            )

        # -------------------------------------------------

        similarity_scores.append(
            max_similarity
        )

        if max_similarity >= SIMILARITY_THRESHOLD:

            matched += 1

    # -----------------------------------------------------

    semantic_fidelity = (
        matched / len(top_features)
    )

    mean_similarity = np.mean(
        similarity_scores
    )

    return (
        semantic_fidelity,
        mean_similarity
    )

# =========================================================
# SEMANTIC FEATURE EXTRACTION
# =========================================================

def extract_semantic_features(
    explanation,
    top_features
):

    sentences = split_sentences(
        explanation
    )

    matched_features = set()

    # -----------------------------------------------------

    for _, row in top_features.iterrows():

        feature = row["Feature"]

        feature_embedding = model.encode(
            feature,
            convert_to_tensor=True
        )

        # -------------------------------------------------

        for sent in sentences:

            sent_embedding = model.encode(
                sent,
                convert_to_tensor=True
            )

            similarity = util.cos_sim(
                feature_embedding,
                sent_embedding
            ).item()

            if similarity >= SIMILARITY_THRESHOLD:

                matched_features.add(
                    feature
                )

                break

    return matched_features

# =========================================================
# SEMANTIC ENTROPY
# =========================================================

def compute_semantic_entropy(
    feature_sets
):

    total_runs = len(feature_sets)

    counter = Counter()

    for fs in feature_sets:

        for f in set(fs):

            counter[f] += 1

    entropy = 0

    # -----------------------------------------------------

    for count in counter.values():

        p = count / total_runs

        entropy -= (
            p * math.log2(p)
        )

    # -----------------------------------------------------

    max_entropy = (
        math.log2(len(counter))
        if len(counter) > 1
        else 1
    )

    normalized_entropy = (
        entropy / max_entropy
    )

    return normalized_entropy

# =========================================================
# RUN S-FEEM
# =========================================================

def run_sfeem_evaluation(

    model_name,
    explanation_folder,
    shap_csv_path

):

    top_features = pd.read_csv(
        shap_csv_path
    ).head(10)

    explanations = load_explanations(
        explanation_folder
    )

    # -----------------------------------------------------

    semantic_scores = []

    similarity_scores = []

    semantic_feature_sets = []

    # -----------------------------------------------------

    for exp in explanations:

        fidelity, similarity = (
            compute_semantic_fidelity(
                exp,
                top_features
            )
        )

        features = (
            extract_semantic_features(
                exp,
                top_features
            )
        )

        semantic_scores.append(
            fidelity
        )

        similarity_scores.append(
            similarity
        )

        semantic_feature_sets.append(
            features
        )

    # -----------------------------------------------------

    avg_semantic_fidelity = np.mean(
        semantic_scores
    )

    avg_similarity = np.mean(
        similarity_scores
    )

    semantic_entropy = (
        compute_semantic_entropy(
            semantic_feature_sets
        )
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    os.makedirs(
        "outputs/sfeem_results",
        exist_ok=True
    )

    results_df = pd.DataFrame({

        "Run": list(
            range(1, len(explanations)+1)
        ),

        "Semantic_Fidelity":
        semantic_scores,

        "Mean_Similarity":
        similarity_scores
    })

    results_df.to_csv(

        f"outputs/sfeem_results/"
        f"{model_name}_sfeem.csv",

        index=False
    )

    # -----------------------------------------------------

    summary = {

        "Model": model_name,

        "Average_Semantic_Fidelity":
        round(avg_semantic_fidelity, 3),

        "Average_BERTScore_Similarity":
        round(avg_similarity, 3),

        "Semantic_Entropy":
        round(semantic_entropy, 3)
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(

        f"outputs/sfeem_results/"
        f"{model_name}_summary.csv",

        index=False
    )

    # -----------------------------------------------------

    print("\n=================================")

    print(f"S-FEEM RESULTS — {model_name}")

    print("=================================")

    print(
        f"Semantic Fidelity: "
        f"{avg_semantic_fidelity:.3f}"
    )

    print(
        f"Mean Similarity: "
        f"{avg_similarity:.3f}"
    )

    print(
        f"Semantic Entropy: "
        f"{semantic_entropy:.3f}"
    )

    print("=================================\n")

# ======================================================
# S-FEEM COMPARISON GRAPH
# ======================================================

    metrics = [

        avg_semantic_fidelity,

        avg_similarity,

        semantic_entropy
    ]

    labels = [

        "Semantic Fidelity",

        "Average Similarity",

        "Semantic Entropy"
    ]

    plt.bar(
        labels,
        metrics
    )

    plt.title(
        f"{model_name.upper()} S-FEEM Results"
    )

    plt.ylim(0, 1)

    plt.savefig(
        f"outputs/sfeem_results/{model_name}_sfeem_graph.png",
        dpi=300
    )

    plt.show()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    SHAP_PATH = (
        "outputs/shap/"
        "shap_feature_importance.csv"
    )

    # -----------------------------------------------------

    run_sfeem_evaluation(

        model_name="llama",

        explanation_folder=
        "outputs/llm_outputs/llama",

        shap_csv_path=SHAP_PATH
    )

    # -----------------------------------------------------

    run_sfeem_evaluation(

        model_name="mistral",

        explanation_folder=
        "outputs/llm_outputs/mistral",

        shap_csv_path=SHAP_PATH
    )