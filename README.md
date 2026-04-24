# 🏦 Predicting Customer Response to Bank Marketing Campaign

An end-to-end **Streamlit** machine learning app that predicts whether a bank customer will subscribe to a term deposit — built on the **ABA Bank Marketing Dataset**.

---

## 🎯 Variable Setup

| Role | Variable | Description |
|------|----------|-------------|
| **Dependent (Target)** | `Y` | Term deposit subscription — `yes` / `no` |
| Independent | `AGE` | Client age (years) |
| Independent | `JOB` | Job type |
| Independent | `MARITAL` | Marital status |
| Independent | `EDUCATION` | Education level |
| Independent | `DEFAULT` | Has credit in default? |
| Independent | `BALANCE` | Avg yearly balance (€) |
| Independent | `HOUSING` | Has housing loan? |
| Independent | `LOAN` | Has personal loan? |
| Independent | `CONTACT` | Contact type |
| Independent | `DAY` | Last contact day of month |
| Independent | `MONTH` | Last contact month |
| Independent | `DURATION` | Last contact duration (seconds) |
| Independent | `CAMPAIGN` | Contacts this campaign |
| Independent | `PDAYS` | Days since last previous contact |
| Independent | `PREVIOUS` | Previous contacts count |
| Independent | `POUTCOME` | Previous campaign outcome |

---

## 🚀 App Features (5 Tabs)

| Tab | Contents |
|-----|----------|
| 📊 Data Overview | Metrics, raw data, stats, variable reference |
| 🔍 EDA | Pie chart, histograms, boxplots, categorical rates, heatmap, month analysis |
| 🤖 Model Training | Feature selection, algorithm choice, train/test split, coefficients / importance |
| 📈 Evaluation | Confusion matrix, ROC-AUC, metrics, classification report, call-reduction business analysis |
| 🔮 Predict | Real-time client form → subscription prediction + probability gauge |

**Algorithms supported:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting

---

## 🛠️ Run Locally (VS Code)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/bank-marketing-predictor.git
cd bank-marketing-predictor

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Add dataset
# Place ABA_bank_marketing.xlsx in the root folder

# 5. Run
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud

1. Push repo to GitHub (include `ABA_bank_marketing.xlsx`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → branch: `main` → Main file: `app.py`
4. Click **Deploy**

---

## 📁 Project Structure

```
bank-marketing-predictor/
├── app.py                      # Main Streamlit application
├── ABA_bank_marketing.xlsx     # Dataset
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md
└── .streamlit/
    └── config.toml             # Theme & server settings
```

---

## 🧰 Tech Stack

`streamlit` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `openpyxl`
