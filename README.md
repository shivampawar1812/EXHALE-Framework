# CLEAR-AI: Clinical Language based Explainable Reasoning AI

An Explainable and LLM-Augmented Machine Learning Framework for Asthma Prediction using NHANES 2011–2012 Data.

---

# Overview

CLEAR-AI-Framework is a healthcare-focused Explainable AI (XAI) research pipeline developed for asthma prediction using the NHANES 2011–2012 dataset.

The framework combines:

* Classical Machine Learning
* Hyperparameter-Tuned Models
* Explainable AI (SHAP)
* Multiple Large Language Models (LLMs)
* Semantic Fidelity and Entropy Evaluation Metircs (S-FEEM)

to create clinically interpretable and trustworthy AI predictions.

The project focuses not only on predictive performance but also on explanation quality, interpretability, and semantic consistency across multiple LLMs.

---

# Research Objectives

The primary goals of this research are:

* Predict asthma diagnosis using NHANES clinical and demographic data
* Compare multiple baseline ML algorithms
* Improve performance using hyperparameter tuning
* Generate patient-level explanations using SHAP
* Convert SHAP outputs into human-readable clinical explanations using multiple LLMs
* Evaluate explanation fidelity using the proposed S-FEEM framework

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

Purpose:
Evaluate the quality, semantic consistency, and fidelity of LLM-generated explanations.

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
| Llama 3.3 70B | 0.750    | 0.792             | 0.510   |
| Mistral Large | 0.840    | 0.804             | 0.268   |

---

## S-FEEM Results (Semantic Fidelity)

| Model         | Semantic Fidelity | Mean Similarity | Semantic Entropy |
| ------------- | ----------------- | --------------- | ---------------- |
| Llama 3.3 70B | 0.830             | 0.605           | 0.365            |
| Mistral Large | 0.990             | 0.642           | 0.490            |

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

# Research Contribution

This framework contributes:

* A reproducible asthma prediction pipeline
* Integration of explainable AI with healthcare ML
* Multi-LLM explanation generation
* A novel semantic fidelity evaluation framework (S-FEEM)

The work aims to improve transparency and trustworthiness in clinical AI systems.

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

AI/ML Research Project — CLEAR-AI-Framework
Manipal University Jaipur
```


