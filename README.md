# Kaggle Titanic - Analysis Environment

## Directory Structure

```
1st_/
├── .venv/                  # Python 3.14 virtual environment
├── data/
│   ├── raw/                # Place Kaggle data here (train.csv, test.csv)
│   └── processed/          # Processed data & charts output
├── notebooks/
│   └── titanic_eda.ipynb   # EDA & Modeling Notebook
├── submissions/            # Submission files output
├── requirements.txt
└── README.md
```

## Setup

### 1. Prepare Data

**Option A: Kaggle CLI**
```powershell
.venv\Scripts\activate
kaggle competitions download -c titanic -p data/raw
cd data/raw; tar -xf titanic.zip
```

**Option B: Manual Download**
Download from https://www.kaggle.com/c/titanic/data
Place `train.csv`, `test.csv`, `gender_submission.csv` into `data/raw/`

### 2. Launch Notebook

```powershell
.venv\Scripts\activate
jupyter notebook notebooks/titanic_eda.ipynb
```

### 3. Submit to Kaggle

```powershell
kaggle competitions submit -c titanic -f submissions/submission.csv -m "first submission"
```

## Rebuild Environment

```powershell
C:\Users\jun\AppData\Local\Programs\Python\Python314\python.exe -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Notes
- Path must be ASCII-only (Japanese characters in path cause scipy DLL errors)
- Python 3.14.6 / pip 26.x
