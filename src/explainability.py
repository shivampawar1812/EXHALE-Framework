import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    "outputs/shap",
    exist_ok=True
)

# =========================================================
# LOAD DATA
# =========================================================

def load_data():

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

    print(
        "Train Shape:",
        X_train.shape
    )

    print(
        "Test Shape :",
        X_test.shape
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )

# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    model = joblib.load(
        "outputs/models/tuned_xgboost.pkl"
    )

    print("\nTuned XGBoost Loaded")

    return model

# =========================================================
# GENERATE SHAP ANALYSIS
# =========================================================

def generate_shap_explanations(

    model,
    X_train,
    X_test
):

    # -----------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    print("SHAP Explainer Created")

    # -----------------------------------------------------

    shap_values = explainer.shap_values(
        X_test
    )

    print("SHAP Values Computed")

    # =====================================================
    # SUMMARY PLOT
    # =====================================================

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_test,

        show=False
    )

    plt.savefig(

        "outputs/shap/"
        "shap_summary_plot.png",

        bbox_inches="tight",

        dpi=300
    )

    plt.close()

    print(
        "Saved: shap_summary_plot.png"
    )

    # =====================================================
    # BAR PLOT
    # =====================================================

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_test,

        plot_type="bar",

        show=False
    )

    plt.savefig(

        "outputs/shap/"
        "shap_bar_plot.png",

        bbox_inches="tight",

        dpi=300
    )

    plt.close()

    print(
        "Saved: shap_bar_plot.png"
    )

    # =====================================================
    # FORCE PLOT
    # =====================================================

    force_plot = shap.force_plot(

        explainer.expected_value,

        shap_values[0],

        X_test.iloc[0],

        matplotlib=False
    )

    shap.save_html(

        "outputs/shap/"
        "shap_force_plot.html",

        force_plot
    )

    print(
        "Saved: shap_force_plot.html"
    )

    # =====================================================
    # DEPENDENCE PLOT
    # =====================================================

    top_feature = X_test.columns[

        abs(shap_values).mean(0).argmax()
    ]

    plt.figure()

    shap.dependence_plot(

        top_feature,

        shap_values,

        X_test,

        show=False
    )

    plt.savefig(

        "outputs/shap/"
        "shap_dependence_plot.png",

        bbox_inches="tight",

        dpi=300
    )

    plt.close()

    print(
        "Saved: shap_dependence_plot.png"
    )

    # =====================================================
    # FEATURE IMPORTANCE CSV
    # =====================================================

    feature_importance = pd.DataFrame({

        "Feature": X_test.columns,

        "Mean_SHAP_Value":
        abs(shap_values).mean(axis=0)
    })

    feature_importance = (
        feature_importance.sort_values(

            by="Mean_SHAP_Value",

            ascending=False
        )
    )

    feature_importance.to_csv(

        "outputs/shap/"
        "shap_feature_importance.csv",

        index=False
    )

    print(
        "Saved: shap_feature_importance.csv"
    )

    # =====================================================
    # TOP FEATURES
    # =====================================================

    print("\n===================================")

    print("TOP 20 IMPORTANT FEATURES")

    print("===================================")

    print(
        feature_importance.head(20)
    )

    print("\n===================================")

    print("SHAP ANALYSIS COMPLETE")

    print("===================================")

    return feature_importance

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    model = load_model()

    generate_shap_explanations(

        model,

        X_train,

        X_test
    )
