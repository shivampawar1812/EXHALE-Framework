import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs("outputs/models", exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(
    "data/processed/train.csv"
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)

X_train = train_df.drop(
    columns=["target"]
)

y_train = train_df["target"]

X_test = test_df.drop(
    columns=["target"]
)

y_test = test_df["target"]

print("Data Loaded")
print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =========================================================
# BASELINE MODELS
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=500
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "SVM": SVC(
        probability=True,
        kernel="rbf",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

# =========================================================
# TRAIN + EVALUATE BASELINE MODELS
# =========================================================

results = []

best_model = None
best_auc = 0

for model_name, model in models.items():

    print("\n===================================")
    print(f"Training: {model_name}")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    # Save results
    results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    # Save model
    model_path = f"outputs/models/{model_name}.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(f"Saved: {model_path}")

    # Track best baseline model
    if auc > best_auc:

        best_auc = auc
        best_model = model_name

# =========================================================
# SAVE BASELINE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC AUC",
    ascending=False
)

results_df.to_csv(
    "outputs/models/baseline_model_results.csv",
    index=False
)

print("\n===================================")
print("BASELINE MODEL RESULTS")
print(results_df)
print("===================================")

print(f"\nBest Baseline Model: {best_model}")
print(f"Best ROC-AUC: {best_auc:.4f}")

# =========================================================
# HYPERPARAMETER TUNING - SVM
# =========================================================

print("\n===================================")
print("SVM HYPERPARAMETER TUNING")
print("===================================")

param_grid_svm = {

    'C': [0.1, 10],

    'gamma': [0.01, 0.001],

    'kernel': ['rbf']
}

svm = SVC(

    probability=True,

    class_weight='balanced',

    random_state=42
)

grid_svm = GridSearchCV(

    svm,

    param_grid_svm,

    cv=3,

    scoring='roc_auc',

    verbose=2,

    n_jobs=-1
)

grid_svm.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_svm.best_params_)

best_svm = SVC(

    C=grid_svm.best_params_['C'],

    gamma=grid_svm.best_params_['gamma'],

    kernel='rbf',

    probability=True,

    class_weight='balanced',

    random_state=42
)

best_svm.fit(X_train, y_train)

# Predictions
y_prob_svm = best_svm.predict_proba(X_test)[:, 1]

y_pred_svm = (y_prob_svm > 0.30).astype(int)

# Metrics
svm_accuracy = accuracy_score(
    y_test,
    y_pred_svm
)

svm_auc = roc_auc_score(
    y_test,
    y_prob_svm
)

print("\nTUNED SVM RESULTS")
print(f"Accuracy : {svm_accuracy:.4f}")
print(f"ROC-AUC  : {svm_auc:.4f}")

# Save tuned SVM
joblib.dump(
    best_svm,
    "outputs/models/tuned_svm.pkl"
)

print("Saved: outputs/models/tuned_svm.pkl")

# =========================================================
# HYPERPARAMETER TUNING - XGBOOST
# =========================================================

print("\n===================================")
print("XGBOOST HYPERPARAMETER TUNING")
print("===================================")

param_grid_xgb = {

    'n_estimators': [200, 300],

    'max_depth': [4, 5, 6],

    'learning_rate': [0.03, 0.05],

    'subsample': [0.8],

    'colsample_bytree': [0.8]
}

xgb = XGBClassifier(

    scale_pos_weight=5,

    eval_metric='logloss',

    random_state=42
)

grid_xgb = GridSearchCV(

    xgb,

    param_grid_xgb,

    cv=3,

    scoring='roc_auc',

    verbose=2,

    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_xgb.best_params_)

best_xgb = XGBClassifier(

    n_estimators=grid_xgb.best_params_['n_estimators'],

    max_depth=grid_xgb.best_params_['max_depth'],

    learning_rate=grid_xgb.best_params_['learning_rate'],

    subsample=grid_xgb.best_params_['subsample'],

    colsample_bytree=grid_xgb.best_params_['colsample_bytree'],

    scale_pos_weight=5,

    eval_metric='logloss',

    random_state=42
)

best_xgb.fit(X_train, y_train)

# Predictions
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

thresholds = [

    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7
]

print("\n===================================")
print("XGBOOST THRESHOLD ANALYSIS")
print("===================================")

for t in thresholds:

    y_pred_xgb = (

        y_prob_xgb >= t

    ).astype(int)

    print(f"\nThreshold: {t}")

    print(

        "Accuracy:",

        accuracy_score(
            y_test,
            y_pred_xgb
        )
    )

    print(

        "Precision:",

        precision_score(
            y_test,
            y_pred_xgb
        )
    )

    print(

        "Recall:",

        recall_score(
            y_test,
            y_pred_xgb
        )
    )

    print(

        "F1 Score:",

        f1_score(
            y_test,
            y_pred_xgb
        )
    )

    print(

        "ROC AUC:",

        roc_auc_score(
            y_test,
            y_prob_xgb
        )
    )

    print("---------------------------")

# =========================================================
# FINAL XGBOOST EVALUATION
# =========================================================

y_pred_xgb = (

    y_prob_xgb >= 0.5

).astype(int)

print("\n===================================")
print("FINAL XGBOOST RESULTS")
print("===================================")

print("Accuracy Score")

print(
    accuracy_score(
        y_test,
        y_pred_xgb
    )
)

print("\nROC AUC Score")

print(
    roc_auc_score(
        y_test,
        y_prob_xgb
    )
)

# Save tuned XGBoost
joblib.dump(
    best_xgb,
    "outputs/models/tuned_xgboost.pkl"
)

print("\nSaved: outputs/models/tuned_xgboost.pkl")

print("\n===================================")
print("MODEL TRAINING COMPLETE")
print("===================================")

# =========================================================
# SAVE TUNED MODEL RESULTS
# =========================================================

tuned_results = pd.DataFrame([

    {
        "Model": "Tuned SVM",
        "Accuracy": svm_accuracy,
        "ROC AUC": svm_auc
    },

    {
        "Model": "Tuned XGBoost",
        "Accuracy": accuracy_score(y_test, y_pred_xgb),
        "ROC AUC": roc_auc_score(y_test, y_prob_xgb)
    }
])

tuned_results.to_csv(

    "outputs/models/tuned_model_results.csv",

    index=False
)

print("\nSaved: tuned_model_results.csv")