# NHANES Dataset Setup

The EXHALE framework uses publicly available NHANES 2011–2012 data released by the CDC.

Due to NHANES data usage and repository size considerations, the raw `.XPT` files are **not included** in this repository and must be downloaded manually.

---

# Folder Structure

Place all downloaded `.XPT` files inside:

```text
data/raw_xpt/
```

Expected structure:

```text
data/
├── raw_xpt/
├── csv/
└── processed/
```

---

# Required NHANES 2011–2012 Files

Download the following files from the official CDC NHANES website:

| Dataset              | File Name    |
| -------------------- | ------------ |
| Demographics         | DEMO_G.XPT   |
| Medical Conditions   | MCQ_G.XPT    |
| Body Measures        | BMX_G.XPT    |
| Smoking              | SMQ_G.XPT    |
| Sleep Disorders      | SLQ_G.XPT    |
| Blood Pressure       | BPQ_G.XPT    |
| Complete Blood Count | CBC_G.XPT    |
| Dietary Interview    | DR1TOT.XPT   |
| Physical Activity    | PAQ_G.XPT    |
| Biochemistry Profile | BIOPRO_G.XPT |
| Respiratory Health   | RDQ_G.XPT    |
| Cotinie Activity     | COTNAL_G.XPT |
| Glycohemoglobin      | GHB.XPT      |

---

# Official CDC NHANES Links

NHANES 2011–2012 Data Portal:

https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011

Example file download format:

https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT

---

# Download Instructions

1. Open the CDC NHANES portal
2. Download all required `.XPT` files listed above
3. Place the files into:

```text
data/raw_xpt/
```

---

# Dataset Processing Workflow

After placing the files:

## Step 1 — Convert XPT Files to CSV

```bash
python src/xpt_to_csv.py
```

Generated CSV files will be stored in:

```text
data/csv/
```

---

## Step 2 — Data Preprocessing

```bash
python src/data_preprocessing.py
```

This script:

* merges NHANES tables
* performs feature engineering
* cleans missing values
* creates the final asthma dataset

Processed outputs are saved in:

```text
data/processed/
```

---

# Final Dataset

The final processed dataset used for asthma prediction is generated automatically during preprocessing and is not committed to the repository.

---

# Important Note

NHANES data is publicly available from the CDC for research and educational purposes.

Please review the official NHANES documentation and terms of use before using the dataset in downstream applications.
