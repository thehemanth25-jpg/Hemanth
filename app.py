import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a3c6e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a3c6e, #2563eb);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a3c6e;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
        margin: 1.5rem 0 0.8rem 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f4ff;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
DEPENDENT_VAR = "Y"
INDEPENDENT_VARS = [
    "AGE", "JOB", "MARITAL", "EDUCATION", "DEFAULT", "BALANCE",
    "HOUSING", "LOAN", "CONTACT", "DAY", "MONTH", "DURATION",
    "CAMPAIGN", "PDAYS", "PREVIOUS", "POUTCOME",
]

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_excel("ABA_bank_marketing.xlsx")
    df.columns = [c.upper() for c in df.columns]
    return df


def encode_and_scale(df, selected_features):
    df_enc = df[selected_features + [DEPENDENT_VAR]].copy()
    le = LabelEncoder()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    X = df_enc[selected_features]
    y = df_enc[DEPENDENT_VAR]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, scaler


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.title("🏦 Control Panel")
    st.divider()

    uploaded = st.file_uploader("Upload your own dataset (.xlsx)", type=["xlsx"])
    st.divider()

    st.markdown("### 🎯 Variable Selection")
    selected_features = st.multiselect(
        "Independent Variables (Features)",
        options=INDEPENDENT_VARS,
        default=["AGE", "BALANCE", "DURATION", "CAMPAIGN", "PDAYS", "PREVIOUS",
                 "JOB", "MARITAL", "EDUCATION", "HOUSING", "LOAN"],
        help="Select one or more features to train the model."
    )
    st.info(f"**Dependent Variable (Target):** `{DEPENDENT_VAR}` — Term Deposit Subscription (yes / no)")
    st.divider()

    st.markdown("### ⚙️ Model Settings")
    model_name = st.selectbox("Choose Algorithm", list(MODELS.keys()))
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    st.divider()
    st.caption("Built with ❤️ using Streamlit & scikit-learn")

# ──────────────────────────────────────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────────────────────────────────────
df = load_data(uploaded)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏦 Bank Marketing Campaign Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether a client will subscribe to a term deposit · ABA Bank Marketing Dataset</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", "🔍 EDA", "🤖 Model Training", "📈 Model Evaluation", "🔮 Predict"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Overview
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Records", f"{df.shape[0]:,}")
    c2.metric("📌 Features", df.shape[1] - 1)
    c3.metric("✅ Subscribed (Yes)", int((df[DEPENDENT_VAR].str.lower() == "yes").sum()))
    c4.metric("❌ Not Subscribed (No)", int((df[DEPENDENT_VAR].str.lower() == "no").sum()))

    st.markdown('<p class="section-title">Raw Data Sample</p>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<p class="section-title">Statistical Summary</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<p class="section-title">Variable Reference</p>', unsafe_allow_html=True)
    var_info = {
        "Variable": INDEPENDENT_VARS + [DEPENDENT_VAR],
        "Type": ["Numeric","Categorical","Categorical","Categorical","Categorical",
                 "Numeric","Categorical","Categorical","Categorical","Numeric",
                 "Categorical","Numeric","Numeric","Numeric","Numeric","Categorical","Categorical"],
        "Role": ["Independent"] * len(INDEPENDENT_VARS) + ["Dependent"],
        "Description": [
            "Client age (years)", "Job type", "Marital status", "Education level",
            "Has credit in default?", "Average yearly balance (€)", "Has housing loan?",
            "Has personal loan?", "Contact communication type", "Last contact day of month",
            "Last contact month", "Last contact duration (seconds)", "Number of contacts this campaign",
            "Days since last contact from previous campaign", "Previous contacts count",
            "Previous campaign outcome", "Target: subscribed to term deposit? (yes/no)"
        ]
    }
    st.dataframe(pd.DataFrame(var_info), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Target distribution
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df[DEPENDENT_VAR].value_counts()
        colors = ["#2563eb", "#f97316"]
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors,
               startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Target Variable Distribution", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x=DEPENDENT_VAR, palette=["#2563eb","#f97316"], ax=ax)
        ax.set_title("Subscription Count", fontweight="bold")
        ax.set_xlabel("Subscribed (Y)")
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Numeric feature distributions
    st.markdown('<p class="section-title">Numeric Feature Distributions by Target</p>', unsafe_allow_html=True)
    numeric_cols = [c for c in INDEPENDENT_VARS if df[c].dtype in ["int64", "float64"]]
    chosen_num = st.selectbox("Select numeric feature", numeric_cols, key="num_feat")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x=chosen_num, hue=DEPENDENT_VAR, kde=True, bins=30,
                 palette={"yes":"#2563eb","no":"#f97316"}, ax=axes[0])
    axes[0].set_title(f"{chosen_num} Distribution by Target")
    sns.boxplot(data=df, x=DEPENDENT_VAR, y=chosen_num,
                palette={"yes":"#2563eb","no":"#f97316"}, ax=axes[1])
    axes[1].set_title(f"{chosen_num} Boxplot by Target")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Categorical feature
    st.markdown('<p class="section-title">Categorical Feature Analysis</p>', unsafe_allow_html=True)
    cat_cols = [c for c in INDEPENDENT_VARS if df[c].dtype == "object"]
    chosen_cat = st.selectbox("Select categorical feature", cat_cols, key="cat_feat")

    fig, ax = plt.subplots(figsize=(12, 4))
    ct = pd.crosstab(df[chosen_cat], df[DEPENDENT_VAR], normalize="index") * 100
    ct.plot(kind="bar", ax=ax, color=["#f97316","#2563eb"], edgecolor="white", width=0.7)
    ax.set_title(f"Subscription Rate by {chosen_cat} (%)", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Subscribed")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Correlation heatmap
    st.markdown('<p class="section-title">Correlation Heatmap (Numeric Features)</p>', unsafe_allow_html=True)
    df_num = df[numeric_cols].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix", fontweight="bold")
    st.pyplot(fig)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Training
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-title">Model Training</p>', unsafe_allow_html=True)

    if not selected_features:
        st.warning("⚠️ Please select at least one independent variable from the sidebar.")
        st.stop()

    X, y, scaler = encode_and_scale(df, selected_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model = MODELS[model_name]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 📌 Selected Configuration")
        st.write(f"- **Algorithm:** {model_name}")
        st.write(f"- **Dependent Variable:** `{DEPENDENT_VAR}` (Term Deposit Subscription)")
        st.write(f"- **Independent Variables ({len(selected_features)}):** {', '.join(selected_features)}")
        st.write(f"- **Training Samples:** {len(X_train):,}")
        st.write(f"- **Test Samples:** {len(X_test):,}")

    with col2:
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_name}..."):
                model.fit(X_train, y_train)
                st.session_state["model"] = model
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.session_state["X_train"] = X_train
                st.session_state["y_train"] = y_train
                st.session_state["model_name"] = model_name
                st.session_state["selected_features"] = selected_features
                st.session_state["scaler"] = scaler
                st.success(f"✅ {model_name} trained successfully!")

    if "model" in st.session_state:
        m = st.session_state["model"]
        X_te = st.session_state["X_test"]
        y_te = st.session_state["y_test"]
        X_tr = st.session_state["X_train"]
        y_tr = st.session_state["y_train"]

        acc_train = accuracy_score(y_tr, m.predict(X_tr))
        acc_test  = accuracy_score(y_te, m.predict(X_te))

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Train Accuracy", f"{acc_train:.2%}")
        c2.metric("🧪 Test Accuracy",  f"{acc_test:.2%}")
        c3.metric("📉 Overfitting Gap", f"{abs(acc_train - acc_test):.2%}")

        # Feature importance (tree-based models)
        if hasattr(m, "feature_importances_"):
            st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)
            fi = pd.Series(m.feature_importances_, index=selected_features).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4, len(fi) * 0.35)))
            fi.plot(kind="barh", ax=ax, color="#2563eb", edgecolor="white")
            ax.set_title("Feature Importance", fontweight="bold")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)
            plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Evaluation
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-title">Model Evaluation</p>', unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.info("👈 Train a model first in the **Model Training** tab.")
    else:
        m   = st.session_state["model"]
        X_te = st.session_state["X_test"]
        y_te = st.session_state["y_test"]
        y_pred = m.predict(X_te)

        col1, col2 = st.columns(2)

        # Confusion Matrix
        with col1:
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_te, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=m.classes_)
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Confusion Matrix", fontweight="bold")
            st.pyplot(fig)
            plt.close()

        # ROC Curve (if predict_proba available)
        with col2:
            st.markdown("#### ROC Curve")
            if hasattr(m, "predict_proba"):
                le_temp = LabelEncoder()
                y_te_bin = le_temp.fit_transform(y_te)
                y_prob = m.predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_te_bin, y_prob)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve", fontweight="bold")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("ROC curve not available for this model.")

        # Classification Report
        st.markdown('<p class="section-title">Classification Report</p>', unsafe_allow_html=True)
        report = classification_report(y_te, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}").background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                     use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Predict New Client
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="section-title">Predict for a New Client</p>', unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.info("👈 Train a model first in the **Model Training** tab.")
    else:
        feats = st.session_state["selected_features"]
        scaler_obj = st.session_state["scaler"]
        m = st.session_state["model"]

        st.markdown("Fill in the client details below:")

        input_data = {}
        cols = st.columns(3)

        field_cfg = {
            "AGE":       {"type": "num", "min": 18, "max": 95, "default": 35},
            "BALANCE":   {"type": "num", "min": -8000, "max": 100000, "default": 1000},
            "DURATION":  {"type": "num", "min": 0, "max": 5000, "default": 200},
            "CAMPAIGN":  {"type": "num", "min": 1, "max": 50, "default": 2},
            "PDAYS":     {"type": "num", "min": -1, "max": 900, "default": -1},
            "PREVIOUS":  {"type": "num", "min": 0, "max": 50, "default": 0},
            "DAY":       {"type": "num", "min": 1, "max": 31, "default": 15},
            "JOB":       {"type": "cat", "options": ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"]},
            "MARITAL":   {"type": "cat", "options": ["divorced","married","single"]},
            "EDUCATION": {"type": "cat", "options": ["primary","secondary","tertiary","unknown"]},
            "DEFAULT":   {"type": "cat", "options": ["no","yes"]},
            "HOUSING":   {"type": "cat", "options": ["no","yes"]},
            "LOAN":      {"type": "cat", "options": ["no","yes"]},
            "CONTACT":   {"type": "cat", "options": ["cellular","telephone","unknown"]},
            "MONTH":     {"type": "cat", "options": ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]},
            "POUTCOME":  {"type": "cat", "options": ["failure","other","success","unknown"]},
        }

        for i, feat in enumerate(feats):
            cfg = field_cfg.get(feat, {"type": "num", "min": 0, "max": 1000, "default": 0})
            with cols[i % 3]:
                if cfg["type"] == "num":
                    input_data[feat] = st.number_input(feat, min_value=cfg["min"], max_value=cfg["max"], value=cfg["default"])
                else:
                    input_data[feat] = st.selectbox(feat, cfg["options"])

        if st.button("🔮 Predict Subscription", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            for col in input_df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    input_df[col] = 0
            input_scaled = pd.DataFrame(scaler_obj.transform(input_df[feats]), columns=feats)
            prediction = m.predict(input_scaled)[0]
            prob = m.predict_proba(input_scaled)[0] if hasattr(m, "predict_proba") else None

            st.divider()
            if str(prediction).lower() == "yes":
                st.success(f"✅ **Prediction: YES** — This client is likely to subscribe to the term deposit!")
            else:
                st.error(f"❌ **Prediction: NO** — This client is unlikely to subscribe to the term deposit.")

            if prob is not None:
                classes = list(m.classes_)
                prob_dict = dict(zip(classes, prob))
                c1, c2 = st.columns(2)
                c1.metric("Probability: Yes", f"{prob_dict.get('yes', prob_dict.get('YES', 0)):.2%}")
                c2.metric("Probability: No",  f"{prob_dict.get('no', prob_dict.get('NO', 0)):.2%}")
