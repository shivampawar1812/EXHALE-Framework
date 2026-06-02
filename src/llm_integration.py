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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

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

shap_df = pd.read_csv(
    "outputs/shap/shap_feature_importance.csv"
)

top_features = shap_df.head(10)

# =========================================================
# BUILD PROMPT
# =========================================================

feature_text = ""

for _, row in top_features.iterrows():

    feature_text += (
        f"{row['Feature']} "
        f"(importance={row['Mean_SHAP_Value']:.4f})\n"
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

print("PROMPT CREATED")

# =========================================================
# SAVE PROMPT
# =========================================================

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

groq_client = Groq(
    api_key=GROQ_API_KEY
)

mistral_client = Mistral(
    api_key=MISTRAL_API_KEY
)

# =========================================================
# GENERATE LLAMA OUTPUTS
# =========================================================

print("\nGenerating LLAMA Outputs...")

for run in range(NUM_RUNS):

    try:

        print(f"LLAMA Run {run+1}/{NUM_RUNS}")

        response = groq_client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            temperature=TEMPERATURE,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        llama_output = response.choices[0].message.content

        output_path = (
            f"outputs/llm_outputs/llama/"
            f"llama_run_{run+1}.txt"
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

        print(f"LLAMA Run {run+1} Error: {e}")

# =========================================================
# GENERATE MISTRAL OUTPUTS
# =========================================================

print("\nGenerating MISTRAL Outputs...")

for run in range(NUM_RUNS):

    try:

        print(f"MISTRAL Run {run+1}/{NUM_RUNS}")

        response = mistral_client.chat.complete(

            model="mistral-large-latest",

            temperature=TEMPERATURE,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        mistral_output = response.choices[0].message.content

        output_path = (
            f"outputs/llm_outputs/mistral/"
            f"mistral_run_{run+1}.txt"
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

        print(f"MISTRAL Run {run+1} Error: {e}")

# =========================================================
# COMPLETE
# =========================================================

print("\n===================================")
print("MULTI-RUN LLM GENERATION COMPLETE")
print("===================================")

