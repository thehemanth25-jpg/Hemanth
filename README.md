# 🏦 Bank Marketing Campaign Predictor

A full-featured **Streamlit** machine learning dashboard to predict whether a bank client will subscribe to a term deposit — built on the **ABA Bank Marketing Dataset**.

---

## 📌 Variable Overview

| Role | Variable | Description |
|------|----------|-------------|
| **Dependent (Target)** | `Y` | Did the client subscribe? (`yes` / `no`) |
| Independent | `AGE` | Client age in years |
| Independent | `JOB` | Type of job |
| Independent | `MARITAL` | Marital status |
| Independent | `EDUCATION` | Education level |
| Independent | `DEFAULT` | Has credit in default? |
| Independent | `BALANCE` | Avg yearly balance (€) |
| Independent | `HOUSING` | Has housing loan? |
| Independent | `LOAN` | Has personal loan? |
| Independent | `CONTACT` | Contact communication type |
| Independent | `DAY` | Last contact day of month |
| Independent | `MONTH` | Last contact month |
| Independent | `DURATION` | Last contact duration (seconds) |
| Independent | `CAMPAIGN` | Number of contacts this campaign |
| Independent | `PDAYS` | Days since last contact (−1 = never) |
| Independent | `PREVIOUS` | Previous contacts count |
| Independent | `POUTCOME` | Previous campaign outcome |

---

## 🚀 Features

- **Data Overview** — raw data preview, statistics, and variable reference table  
- **EDA** — target distribution, numeric distributions, categorical breakdowns, correlation heatmap  
- **Model Training** — choose algorithm, select features, set test split  
- **Model Evaluation** — confusion matrix, ROC curve, AUC score, classification report  
- **Predict** — interactive form to predict for a new client  

### Supported Algorithms
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

---

## 🛠️ Local Setup (VS Code)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-marketing-predictor.git
cd bank-marketing-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `ABA_bank_marketing.xlsx` in the root folder (same level as `app.py`).

### 5. Run the app
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch (`main`), and set **Main file** to `app.py`.
4. Upload `ABA_bank_marketing.xlsx` via the **Secrets / Files** section or use Streamlit's file uploader in the sidebar.
5. Click **Deploy**.

---

## 📁 Project Structure

```
bank-marketing-predictor/
├── app.py                    # Main Streamlit application
├── ABA_bank_marketing.xlsx   # Dataset (add manually)
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from Git
└── README.md                 # This file
```

---

## 📊 Dataset

- **Source:** ABA Bank Marketing Dataset  
- **Records:** 4,521  
- **Features:** 16 independent variables + 1 target variable  
- **Task:** Binary classification (`yes` / `no`)  

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | Machine learning |
| `openpyxl` | Excel file reading |

---

## 📄 License
MIT License — free to use and modify.
