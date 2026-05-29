import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD DATASET
# =========================================================

DATA_PATH = "data/csv/NHANES_2011_12_asthma_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded")
print(df.shape)


# =========================================================
# TARGET COLUMN
# =========================================================

TARGET = "asthma_diagnosis"

# =========================================================
# REMOVE INVALID TARGET VALUES
# =========================================================

df = df[df[TARGET].isin([1, 2])]

# Convert:
# 1 = Yes
# 2 = No

df[TARGET] = df[TARGET].map({
    1: 1,
    2: 0
})
print(df[TARGET].value_counts())

# =========================================================
# DROP ID COLUMN
# =========================================================

if "SEQN" in df.columns:
    df.drop(columns=["SEQN"], inplace=True)


print(df.shape)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================
# KNN IMPUTATION 

numeric_cols = df.select_dtypes(include=np.number).columns
knn_imputer = KNNImputer(
    
    n_neighbors=10,
    
    weights='uniform'
)
df[numeric_cols] = knn_imputer.fit_transform(
    
    df[numeric_cols]
)

print(df.isnull().sum())


# =========================================================
# EXTRACTED FEATURES
# =========================================================
df['waist_height_ratio'] = (
    df['waist_circumference_cm'] /
    df['height_cm']
)

# Obesity feature

df['obese'] = (
    df['bmi'] > 30
).astype(int)

# Smoking intensity

df['smoking_intensity'] = (
    df['serum_cotinine'] *
    df['current_smoking_status']
)

print(df.head())


# =========================================================
# HANDLE CATEGORICAL COLUMNS
# =========================================================

categorical_cols = [

    'gender',

    'race_ethnicity',

    'education_level',

    'wheezing_past_year',

    'asthma_attack_past_year',

    'hay_fever',

    'eczema',

    'chronic_bronchitis',

    'emphysema',

    'family_history_asthma',

    'ever_smoked_100_cigarettes',

    'current_smoking_status',

    'sleep_disorder',

    'vigorous_work_activity',

    'moderate_work_activity',

    'vigorous_recreational_activity',

    'moderate_recreational_activity',

    'obese'
]


df = pd.get_dummies(

    df,

    columns=categorical_cols,

    drop_first=True
)

print(df.shape)


# =========================================================
# SPLIT FEATURES + TARGET
# =========================================================

X = df.drop(columns=[TARGET])

y = df[TARGET]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42
)

print(X_train.shape)
print(X_test.shape)


# =========================================================
# SCALING
# =========================================================

continuous_cols = [

    'age_years',

    'household_size',

    'income_poverty_ratio',

    'episodes_wheezing_year',

    'woke_wheezing_year',

    'white_blood_cell_count',

    'hemoglobin',

    'eosinophil_percent',

    'neutrophil_percent',

    'lymphocyte_percent',

    'platelet_count',

    'weight_kg',
    'height_cm',

    'bmi',

    'waist_circumference_cm',

    'sleep_hours',

    'serum_cotinine',

    'daily_energy_kcal',

    'protein_g',

    'carbohydrate_g',

    'total_fat_g',

    'polyunsaturated_fat_g',

    'sodium_mg',

    'fiber_g',
    'sedentary_minutes_day',

    'creatinine_mg_dl',

    'alt_enzyme',

    'ast_enzyme',

    'uric_acid',

    'hba1c_percent',

    'waist_height_ratio',

    'smoking_intensity'
]

scaler = StandardScaler()

X_train[continuous_cols] = scaler.fit_transform(
    X_train[continuous_cols]
)

X_test[continuous_cols] = scaler.transform(
    X_test[continuous_cols]
)

# =========================================================
# SAVE PREPROCESSED DATA
# =========================================================

train_df = pd.DataFrame(X_train)
train_df["target"] = y_train.values

test_df = pd.DataFrame(X_test)
test_df["target"] = y_test.values

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)

print("\n===================================")
print("PREPROCESSING COMPLETE")
print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)
print("===================================")


print(y_train.value_counts())