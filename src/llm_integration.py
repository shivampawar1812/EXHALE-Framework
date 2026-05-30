import os
import pandas as pd
from dotenv import load_dotenv

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    "outputs/llm_outputs",
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
# LLAMA VIA GROQ
# =========================================================

try:

    from groq import Groq

    client = Groq(
        api_key=GROQ_API_KEY
    )

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    llama_output = response.choices[0].message.content

    with open(
        "outputs/llm_outputs/llama_output.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(llama_output)

    print("Llama Output Saved")

except Exception as e:

    print(f"Llama Error: {e}")

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
# COMPLETE
# =========================================================

print("\n===================================")
print("LLM EXPLANATION GENERATION COMPLETE")
print("===================================")

