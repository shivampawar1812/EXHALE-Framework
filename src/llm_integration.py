import os
import time
import pandas as pd

from dotenv import load_dotenv

from mistralai import Mistral
from groq import Groq

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

MISTRAL_API_KEY = os.getenv(
    "MISTRAL_API_KEY"
)

# =========================================================
# CONFIG
# =========================================================

NUM_RUNS = 10

TEMPERATURE = 0.7

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

os.makedirs(

    "outputs/llm_outputs/llama",

    exist_ok=True
)

os.makedirs(

    "outputs/llm_outputs/mistral",

    exist_ok=True
)

# =========================================================
# LOAD SHAP FEATURES
# =========================================================

def load_top_features():

    shap_df = pd.read_csv(

        "outputs/shap/"
        "shap_feature_importance.csv"
    )

    top_features = shap_df.head(10)

    return top_features

# =========================================================
# BUILD PROMPT
# =========================================================

def build_prompt(top_features):

    feature_text = ""

    for _, row in top_features.iterrows():

        feature_text += (

            f"{row['Feature']} "

            f"(importance="
            f"{row['Mean_SHAP_Value']:.4f})\n"
        )

    prompt = f"""
You are a medical AI explanation system.

An XGBoost model predicted asthma risk.

Top SHAP features:

{feature_text}

Generate:
1. Clinical explanation
2. Patient-friendly explanation
3. Risk interpretation
4. Preventive recommendations

Keep the explanation medically meaningful.
"""

    return prompt

# =========================================================
# SAVE PROMPT
# =========================================================

def save_prompt(prompt):

    with open(

        "outputs/llm_outputs/prompt.txt",

        "w",

        encoding="utf-8"
    ) as f:

        f.write(prompt)

    print("Prompt Saved")

# =========================================================
# INITIALIZE CLIENTS
# =========================================================

def initialize_clients():

    groq_client = Groq(

        api_key=GROQ_API_KEY
    )

    mistral_client = Mistral(

        api_key=MISTRAL_API_KEY
    )

    return (
        groq_client,
        mistral_client
    )

# =========================================================
# GENERATE LLAMA OUTPUTS
# =========================================================

def generate_llama_outputs(

    groq_client,
    prompt
):

    print(
        "\nGenerating LLAMA Outputs..."
    )

    for run in range(NUM_RUNS):

        try:

            print(
                f"LLAMA Run "
                f"{run+1}/{NUM_RUNS}"
            )

            response = (
                groq_client
                .chat
                .completions
                .create(

                    model=
                    "llama-3.3-70b-versatile",

                    temperature=
                    TEMPERATURE,

                    messages=[

                        {
                            "role": "user",

                            "content": prompt
                        }
                    ]
                )
            )

            llama_output = (

                response
                .choices[0]
                .message
                .content
            )

            output_path = (

                "outputs/llm_outputs/"
                f"llama/llama_run_"
                f"{run+1}.txt"
            )

            with open(

                output_path,

                "w",

                encoding="utf-8"
            ) as f:

                f.write(llama_output)

            print(f"Saved: {output_path}")

            time.sleep(1)

        except Exception as e:

            print(
                f"LLAMA Run "
                f"{run+1} Error: {e}"
            )

# =========================================================
# GENERATE MISTRAL OUTPUTS
# =========================================================

def generate_mistral_outputs(

    mistral_client,
    prompt
):

    print(
        "\nGenerating MISTRAL Outputs..."
    )

    for run in range(NUM_RUNS):

        try:

            print(
                f"MISTRAL Run "
                f"{run+1}/{NUM_RUNS}"
            )

            response = (
                mistral_client
                .chat
                .complete(

                    model=
                    "mistral-large-latest",

                    temperature=
                    TEMPERATURE,

                    messages=[

                        {
                            "role": "user",

                            "content": prompt
                        }
                    ]
                )
            )

            mistral_output = (

                response
                .choices[0]
                .message
                .content
            )

            output_path = (

                "outputs/llm_outputs/"
                f"mistral/mistral_run_"
                f"{run+1}.txt"
            )

            with open(

                output_path,

                "w",

                encoding="utf-8"
            ) as f:

                f.write(mistral_output)

            print(f"Saved: {output_path}")

            time.sleep(1)

        except Exception as e:

            print(
                f"MISTRAL Run "
                f"{run+1} Error: {e}"
            )

# =========================================================
# MAIN PIPELINE
# =========================================================

def generate_llm_explanations():

    top_features = load_top_features()

    prompt = build_prompt(
        top_features
    )

    print("PROMPT CREATED")

    save_prompt(prompt)

    (
        groq_client,
        mistral_client
    ) = initialize_clients()

    generate_llama_outputs(

        groq_client,
        prompt
    )

    generate_mistral_outputs(

        mistral_client,
        prompt
    )

    print("\n===================================")

    print(
        "MULTI-RUN LLM "
        "GENERATION COMPLETE"
    )

    print("===================================")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    generate_llm_explanations()

