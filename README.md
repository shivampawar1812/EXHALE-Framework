# EXHALE

## EXplainable Healthcare AI with Linguistic Evaluation

An interpretable machine learning and large language model framework for asthma risk prediction, clinical reasoning generation, and explanation fidelity evaluation using NHANES 2011–2012 data.

---

# Overview

EXHALE (EXplainable Healthcare AI with Linguistic Evaluation) is a healthcare-focused explainable AI framework designed to bridge the gap between machine learning predictions and clinically meaningful reasoning.

The framework combines predictive modeling, SHAP-based explainability, large language models (LLMs), and a novel semantic evaluation framework (S-FEEM) to generate, analyze, and validate clinical explanations for asthma risk prediction.

Unlike traditional prediction pipelines that focus solely on classification performance, EXHALE emphasizes interpretability, transparency, and explanation faithfulness, enabling healthcare stakeholders to understand not only what a model predicts, but why it predicts it.

---

# Research Objectives

The primary objectives of EXHALE are:

• Develop interpretable machine learning models for asthma risk prediction using NHANES 2011–2012 data.

• Translate feature-level model explanations into clinically meaningful reasoning using large language models.

• Quantify the fidelity and consistency of LLM-generated explanations.

• Introduce S-FEEM, a semantic evaluation framework for assessing explanation quality beyond lexical overlap.

• Investigate how explanation faithfulness varies across different LLM architectures.

---

# Dataset

Dataset Used:
NHANES 2011–2012 (National Health and Nutrition Examination Survey)

Source:
CDC NHANES Public Dataset

Raw NHANES `.XPT` files are processed and merged to create the final asthma prediction dataset.

---

# Features Used

The framework integrates multi-domain health variables including:

* Demographics
* Respiratory conditions
* Blood biomarkers
* Smoking history
* Sleep behavior
* Physical activity
* Dietary intake
* Biochemistry
* Body measurements

Target Variable:

* Asthma Diagnosis (`MCQ010`)

---

# Project Pipeline

```text
NHANES .XPT Files
        ↓
XPT Parsing + Merging
        ↓
Data Cleaning + Preprocessing
        ↓
Baseline ML Models
        ↓
Hyperparameter Tuning
        ↓
Best Model Selection
        ↓
SHAP Explainability
        ↓
LLM-Based Explanation Generation
        ↓
S-FEEM Evaluation
```

---

# Repository Structure

```text
CLEAR-AI-Framework/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw_xpt/
│   ├── csv/
│   └── processed/
│
├── src/
│   ├── xpt_to_csv.py
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── explainability.py
│   ├── llm_integration.py
│   ├── sfeem.py
│   └── utils.py
│
└── outputs/
    ├── models/
    ├── shap/
    ├── llm_outputs/
    └── sfeem_results/
```

---

# Machine Learning Models

The framework evaluates multiple classical machine learning algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* Naive Bayes
* Support Vector Machine (SVM)
* XGBoost

Model comparison is performed using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Best model selection is based primarily on ROC-AUC score due to the medical classification setting.

---

# Hyperparameter Tuning

Advanced tuning is performed for:

## Tuned SVM

Using:

* GridSearchCV
* ROC-AUC optimization
* Threshold tuning

## Tuned XGBoost

Using:

* GridSearchCV
* Depth optimization
* Learning rate tuning
* Threshold analysis

---

# Explainable AI (XAI)

The framework uses SHAP (SHapley Additive Explanations) to interpret model predictions.

Implemented SHAP visualizations include:

* SHAP Summary Plot
* SHAP Feature Importance Plot
* SHAP Dependence Plot
* SHAP Force Plot

These explanations help identify the most influential clinical and demographic features contributing to asthma prediction.

---

# Multi-LLM Explanation Framework

The framework integrates multiple Large Language Models to transform SHAP outputs into human-readable clinical explanations.

Integrated LLMs:

* Mistral
* Llama

Each LLM receives:

* Prediction outputs
* Top SHAP features
* Feature contributions

and generates:

* Clinical explanations
* Patient-friendly interpretations
* Risk reasoning summaries

---

# FEEM Framework

FEEM (Fidelity Entropy Evaluation Metric) is the lexical evaluation framework implemented in CLEAR-AI for assessing the reliability and consistency of LLM-generated clinical explanations.

Purpose:
Evaluate whether LLM explanations explicitly preserve SHAP-derived feature importance through keyword-level fidelity analysis.

Evaluation Components:

Exact feature fidelity
Weighted fidelity scoring
Entropy-based consistency analysis
Exact match accuracy
Cross-LLM lexical comparison

FEEM serves as the baseline explainability evaluation framework and provides the foundation for the proposed semantic extension, S-FEEM.

---

# S-FEEM Framework

S-FEEM (Semantic Fidelity Entropy Evaluation Metric) is the proposed evaluation framework developed in this research.

S-FEEM was developed to quantitatively assess the quality of LLM-generated explanations. The metric combines four complementary dimensions: semantic similarity (S), feature fidelity (F), consistency (C), and hallucination robustness (H).

The overall score is computed as:

S-FEEM=αS+βF+γC+δH


where α+β+γ+δ=1. Semantic similarity measures agreement between generated and reference explanations, fidelity evaluates alignment with SHAP-important features, consistency quantifies stability across repeated generations, and hallucination robustness penalizes unsupported claims. Higher S-FEEM scores indicate more reliable, faithful, and clinically trustworthy explanations.

Evaluation Components:

* Semantic similarity
* Explanation fidelity
* Entropy-based consistency
* Hallucination detection
* Cross-LLM explanation comparison

This component represents the core research novelty of the framework.

---

# Outputs

Generated outputs include:

## Models

* Baseline ML models
* Tuned SVM
* Tuned XGBoost

## Explainability

* SHAP plots
* Feature importance CSVs

## LLM Outputs

* Multi-LLM explanations
* Clinical reasoning summaries

## Evaluation

* FEEM scores
* S-FEEM scores
* Semantic similarity reports
* Fidelity comparison tables
  
---

# Experimental Results

## Model Performance

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.873    | 0.768   |
| SVM                 | 0.874    | 0.764   |
| Random Forest       | 0.871    | 0.779   |
| XGBoost             | 0.867    | 0.756   |
| Tuned SVM           | 0.849    | 0.771   |
| Tuned XGBoost       | 0.827    | 0.790   |

The tuned XGBoost model achieved the highest ROC-AUC score and was selected as the primary prediction model for explainability and LLM-based reasoning generation.

---

## FEEM Results (Lexical Fidelity)

| Model         | Fidelity | Weighted Fidelity | Entropy |
| ------------- | -------- | ----------------- | ------- |
| Llama 3.3 70B | 0.610    | 0.586             | 0.752   |
| Mistral Large | 0.720    | 0.605             | 0.167   |


## S-FEEM Results (Semantic Fidelity)

| Model         | Semantic Fidelity | Mean Similarity | Semantic Entropy |
| ------------- | ----------------- | --------------- | ---------------- |
| Llama 3.3 70B | 0.800             | 0.602           | 0.734            |
| Mistral Large | 0.960             | 0.652           | 0.482            |

---

## Key Findings

* S-FEEM consistently produced higher fidelity scores than keyword-based FEEM, demonstrating the importance of semantic evaluation for LLM-generated clinical explanations.
* Mistral generated highly aligned semantic explanations, while Llama produced more lexically diverse reasoning patterns.
* Entropy analysis revealed measurable differences in explanation consistency across LLM architectures.
* The proposed S-FEEM framework effectively captures semantically equivalent medical reasoning missed by traditional lexical overlap metrics.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/shivampawar1812/CLEAR-AI-Framework.git
cd CLEAR-AI-Framework
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Execution Workflow

## Step 1 — Convert NHANES XPT Files

```bash
python src/xpt_to_csv.py
```

## Step 2 — Data Preprocessing

```bash
python src/data_preprocessing.py
```

## Step 3 — Train Models

```bash
python src/models.py
```

## Step 4 — Generate SHAP Explanations

```bash
python src/explainability.py
```

## Step 5 — Generate LLM Explanations

```bash
python src/llm_integration.py
```

## Step 6 — Run S-FEEM Evaluation

```bash
python src/sfeem.py
```

---

# Research Contributions

EXHALE contributes:

• An interpretable machine learning framework for asthma risk prediction using nationally representative healthcare data.

• A clinical reasoning generation pipeline that transforms SHAP-based explanations into human-readable medical narratives using multiple LLMs.

• FEEM, a lexical fidelity evaluation framework for assessing explanation faithfulness.

• S-FEEM, a semantic extension that evaluates explanation fidelity, consistency, and hallucination robustness.

• A reproducible methodology for studying trustworthiness and interpretability in LLM-augmented healthcare AI systems.

---

# Future Work

Potential extensions include:

* External dataset validation
* Real-time clinical dashboards
* RAG-enhanced medical explanations
* Federated healthcare learning
* Temporal patient modeling
* Multi-modal health data integration

---

# Disclaimer

This framework is intended solely for research and educational purposes.

It is not designed for direct clinical deployment or medical diagnosis.

---

# License

MIT License

---


# Authors

- Shivam Pawar
- Harshita Dixit

AI/ML Research Project — EXHALE-Framework

Manipal University Jaipur
```


