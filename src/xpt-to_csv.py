import os
import pandas as pd
import pyreadstat

# =========================================================
# PATHS
# =========================================================

RAW_XPT_DIR = "data/raw_xpt"
CSV_DIR = "data/csv"

os.makedirs(CSV_DIR, exist_ok=True)

# =========================================================
# NHANES FILES + VARIABLES
# =========================================================

FILES = {

    "DEMO_G": {

        "RIDAGEYR": "age_years",
        "RIAGENDR": "gender",
        "RIDRETH3": "race_ethnicity",
        "DMDEDUC2": "education_level",
        "DMDHHSIZ": "household_size",
        "INDFMPIR": "income_poverty_ratio",
    },

    "MCQ_G": {

        "MCQ010": "asthma_diagnosis",
        "MCQ160L": "wheezing_past_year",
        "MCQ160M": "asthma_attack_past_year",
        "MCQ160E": "hay_fever",
        "MCQ160F": "eczema",
        "MCQ160K": "chronic_bronchitis",
        "MCQ160G": "emphysema",
        "MCQ160O": "copd",
        "MCQ300A": "family_history_asthma",
    },

    "RDQ_G": {

        "RDQ070": "episodes_wheezing_year",
        "RDQ080": "woke_wheezing_year",
        "RDQ090": "woke_chest_tightness",
        "RDQ100": "woke_breathlessness",
        "RDQ110": "asthma_attacks_year",
        "RDQ134": "emergency_asthma_visits",
        "RDD140Q": "dry_cough_night",
    },

    "CBC_G": {

        "LBXWBCSI": "white_blood_cell_count",
        "LBXHGB": "hemoglobin",
        "LBXEOPCT": "eosinophil_percent",
        "LBXNEPCT": "neutrophil_percent",
        "LBXLYPCT": "lymphocyte_percent",
        "LBXPLTSI": "platelet_count",
    },

    "BMX_G": {

        "BMXWT": "weight_kg",
        "BMXHT": "height_cm",
        "BMXBMI": "bmi",
        "BMXWAIST": "waist_circumference_cm",
    },

    "SMQ_G": {

        "SMQ020": "ever_smoked_100_cigarettes",
        "SMQ040": "current_smoking_status",
        "SMD460": "smokers_in_household",
        "SMD470": "cigarettes_per_day",
    },

    "SLQ_G": {

        "SLD010H": "sleep_hours",
        "SLQ030": "snoring_frequency",
        "SLQ050": "sleep_disorder",
    },

    "COTNAL_G": {

        "LBXCOT": "serum_cotinine",
    },

    "DR1TOT_G": {

        "DR1TKCAL": "daily_energy_kcal",
        "DR1TPROT": "protein_g",
        "DR1TCARB": "carbohydrate_g",
        "DR1TTFAT": "total_fat_g",
        "DR1TPFAT": "polyunsaturated_fat_g",
        "DR1TSODI": "sodium_mg",
        "DR1TFIBE": "fiber_g",
    },

    "PAQ_G": {

        "PAQ605": "vigorous_work_activity",
        "PAQ620": "moderate_work_activity",
        "PAQ650": "vigorous_recreational_activity",
        "PAQ665": "moderate_recreational_activity",
        "PAD680": "sedentary_minutes_day",
    },

    "BIOPRO_G": {

        "LBXSGLU": "glucose_mg_dl",
        "LBXSCR": "creatinine_mg_dl",
        "LBXSATSI": "alt_enzyme",
        "LBXSASSI": "ast_enzyme",
        "LBXSUA": "uric_acid",
    },

    "GHB_G": {

        "LBXGH": "hba1c_percent",
    },
}

# =========================================================
# LOAD XPT FILE
# =========================================================

def load_xpt(file_path, selected_columns):

    print(f"Loading: {file_path}")

    df, meta = pyreadstat.read_xport(file_path)

    cols = ["SEQN"] + selected_columns

    available_cols = [c for c in cols if c in df.columns]

    df = df[available_cols]

    return df


# =========================================================
# BUILD FINAL DATASET
# =========================================================

def build_dataset():

    merged_df = None

    for file_name, variable_map in FILES.items():

        file_path = os.path.join(
            RAW_XPT_DIR,
            f"{file_name}.xpt"
        )

        if not os.path.exists(file_path):

            print(f"Missing File: {file_path}")
            continue

        nhanes_columns = list(variable_map.keys())

        try:

            temp_df = load_xpt(
                file_path,
                nhanes_columns
            )

        except Exception as e:

            print(f"Error loading {file_path}: {e}")
            continue

        # Rename columns
        temp_df.rename(
            columns=variable_map,
            inplace=True
        )

        # Merge
        if merged_df is None:

            merged_df = temp_df

        else:

            merged_df = pd.merge(
                merged_df,
                temp_df,
                on="SEQN",
                how="outer"
            )

        print(f"Current Shape: {merged_df.shape}")

    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    missing_codes = [
        7, 9,
        77, 99,
        777, 999,
        7777, 9999
    ]

    for code in missing_codes:
        merged_df = merged_df.mask(merged_df == code, pd.NA)
    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    merged_df.drop_duplicates(
        subset=["SEQN"],
        inplace=True
    )

    # =====================================================
    # REMOVE MISSING TARGET
    # =====================================================

    merged_df = merged_df[
        merged_df["asthma_diagnosis"].notna()
    ]

    # =====================================================
    # SAVE CSV
    # =====================================================

    output_path = os.path.join(
        CSV_DIR,
        "NHANES_2011_12_asthma_dataset.csv"
    )

    merged_df.to_csv(
        output_path,
        index=False
    )

    print("\n================================================")
    print(f"DATASET SAVED: {output_path}")
    print(f"FINAL SHAPE : {merged_df.shape}")
    print("================================================")

    return merged_df


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("\nBUILDING DATASET...\n")

    df = build_dataset()

    print("\nDONE!")