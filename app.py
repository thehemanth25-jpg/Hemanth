import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header{font-size:2.2rem;font-weight:700;color:#1a3c6e;text-align:center;margin-bottom:4px;}
    .sub-header{font-size:1rem;color:#555;text-align:center;margin-bottom:18px;}
    .section-title{font-size:1.25rem;font-weight:600;color:#1a3c6e;
                   border-left:4px solid #2563eb;padding-left:10px;margin:18px 0 10px 0;}
    div[data-testid="stSidebar"]{background-color:#f0f4ff;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
TARGET = "Y"
FEATURES = [
    "AGE","JOB","MARITAL","EDUCATION","DEFAULT","BALANCE",
    "HOUSING","LOAN","CONTACT","DAY","MONTH","DURATION",
    "CAMPAIGN","PDAYS","PREVIOUS","POUTCOME",
]
# Numeric features (dtype kind i/f)
NUM_FEATS  = ["AGE","BALANCE","DAY","DURATION","CAMPAIGN","PDAYS","PREVIOUS"]
# Categorical features
CAT_FEATS  = ["JOB","MARITAL","EDUCATION","DEFAULT","HOUSING","LOAN",
              "CONTACT","MONTH","POUTCOME"]

PALETTE = {"yes": "#2563eb", "no": "#f97316"}

MODELS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# Categorical field options for Predict tab
CAT_OPTIONS = {
    "JOB":       ["admin.","blue-collar","entrepreneur","housemaid","management",
                  "retired","self-employed","services","student","technician",
                  "unemployed","unknown"],
    "MARITAL":   ["divorced","married","single"],
    "EDUCATION": ["primary","secondary","tertiary","unknown"],
    "DEFAULT":   ["no","yes"],
    "HOUSING":   ["no","yes"],
    "LOAN":      ["no","yes"],
    "CONTACT":   ["cellular","telephone","unknown"],
    "MONTH":     ["jan","feb","mar","apr","may","jun",
                  "jul","aug","sep","oct","nov","dec"],
    "POUTCOME":  ["failure","other","success","unknown"],
}
NUM_CFG = {
    "AGE":      {"min": 18,    "max": 95,     "default": 35},
    "BALANCE":  {"min": -8000, "max": 102000, "default": 1000},
    "DAY":      {"min": 1,     "max": 31,     "default": 15},
    "DURATION": {"min": 0,     "max": 5000,   "default": 200},
    "CAMPAIGN": {"min": 1,     "max": 63,     "default": 2},
    "PDAYS":    {"min": -1,    "max": 871,    "default": -1},
    "PREVIOUS": {"min": 0,     "max": 275,    "default": 0},
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING  — cached so it only runs once
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_obj=None):
    if file_obj is not None:
        df = pd.read_excel(file_obj)
    else:
        df = pd.read_excel("ABA_bank_marketing.xlsx")
    df.columns = [c.upper() for c in df.columns]
    # Normalise Y to lowercase so comparisons are consistent
    df[TARGET] = df[TARGET].astype(str).str.lower().str.strip()
    return df

# ─────────────────────────────────────────────────────────────
# PREPROCESSING  — encode & scale for a given feature list
# Returns X_scaled (DataFrame), y (int Series), fitted scaler,
#         fitted label-encoders dict
# ─────────────────────────────────────────────────────────────
def preprocess(df, selected_features):
    df_work = df[selected_features + [TARGET]].copy()

    # Encode categoricals — keep one LE per column so Predict tab can reuse
    encoders = {}
    for col in selected_features:
        if col in CAT_FEATS:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))
            encoders[col] = le

    # Encode target: yes→1, no→0
    le_y = LabelEncoder()
    df_work[TARGET] = le_y.fit_transform(df_work[TARGET].astype(str))

    X = df_work[selected_features].astype(float)
    y = df_work[TARGET]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)

    return X_scaled, y, scaler, encoders, le_y

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=68)
    st.title("🏦 Control Panel")
    st.divider()

    uploaded = st.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])
    st.divider()

    st.markdown("### 🎯 Variable Selection")
    selected_features = st.multiselect(
        "Independent Variables",
        options=FEATURES,
        default=["AGE","BALANCE","DURATION","CAMPAIGN","PDAYS","PREVIOUS",
                 "JOB","MARITAL","EDUCATION","HOUSING","LOAN"],
    )
    st.info(f"**Target (Dependent):** `{TARGET}` — Term Deposit Subscription (yes / no)")
    st.divider()

    st.markdown("### ⚙️ Model Settings")
    model_name = st.selectbox("Algorithm", list(MODELS.keys()))
    test_size  = st.slider("Test Set Size (%)", 10, 40, 30) / 100
    st.divider()
    st.caption("Built with ❤️ using Streamlit & scikit-learn")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = load_data(uploaded)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏦 Bank Marketing Campaign Predictor</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicting Customer Response to Bank Marketing Campaign · ABA Dataset</p>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Data Overview", "🔍 EDA", "🤖 Model Training", "📈 Evaluation", "🔮 Predict"]
)

# ═══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    yes_count = int((df[TARGET] == "yes").sum())
    no_count  = int((df[TARGET] == "no").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Records",       f"{df.shape[0]:,}")
    c2.metric("📌 Total Features",      df.shape[1] - 1)
    c3.metric("✅ Subscribed (Yes)",    yes_count)
    c4.metric("❌ Not Subscribed (No)", no_count)

    st.markdown('<p class="section-title">Raw Data Sample (First 20 Rows)</p>',
                unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<p class="section-title">Statistical Summary</p>', unsafe_allow_html=True)
    # Only numeric cols have meaningful describe; format safely
    desc = df.describe()
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<p class="section-title">Variable Reference Table</p>',
                unsafe_allow_html=True)
    var_df = pd.DataFrame({
        "Variable":    FEATURES + [TARGET],
        "Type":        ["Numeric","Categorical","Categorical","Categorical","Categorical",
                        "Numeric","Categorical","Categorical","Categorical","Numeric",
                        "Categorical","Numeric","Numeric","Numeric","Numeric",
                        "Categorical","Categorical"],
        "Role":        ["Independent"] * len(FEATURES) + ["Dependent"],
        "Description": [
            "Client age (years)",
            "Job type",
            "Marital status",
            "Education level",
            "Has credit in default?",
            "Average yearly balance (€)",
            "Has housing loan?",
            "Has personal loan?",
            "Contact communication type",
            "Last contact day of month",
            "Last contact month",
            "Last contact duration (seconds)",
            "Number of contacts this campaign",
            "Days since last contact (-1 = never)",
            "Previous contacts count",
            "Previous campaign outcome",
            "Subscribed to term deposit? (yes/no)",
        ]
    })
    st.dataframe(var_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Target Variable Distribution</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        counts = df[TARGET].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values,
               labels=counts.index,
               autopct="%1.1f%%",
               colors=["#f97316", "#2563eb"],
               startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Subscription Distribution", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        # FIX: use hue= instead of palette= alone (seaborn 0.13 deprecation)
        sns.countplot(data=df, x=TARGET, hue=TARGET,
                      palette=PALETTE, ax=ax, legend=False)
        ax.set_title("Subscription Count", fontweight="bold")
        ax.set_xlabel("Subscribed (Y)")
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height()):,}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Numeric distributions ──────────────────────────────────
    st.markdown('<p class="section-title">Numeric Feature Distribution by Target</p>',
                unsafe_allow_html=True)

    # Only show numeric features that actually exist in df
    available_num = [c for c in NUM_FEATS if c in df.columns]
    chosen_num = st.selectbox("Select numeric feature", available_num, key="num_sel")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(data=df, x=chosen_num, hue=TARGET,
                 kde=True, bins=30, palette=PALETTE, ax=axes[0])
    axes[0].set_title(f"{chosen_num} — Histogram by Target", fontweight="bold")

    # FIX: boxplot with hue to avoid deprecation
    sns.boxplot(data=df, x=TARGET, y=chosen_num, hue=TARGET,
                palette=PALETTE, ax=axes[1], legend=False)
    axes[1].set_title(f"{chosen_num} — Boxplot by Target", fontweight="bold")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Categorical feature ────────────────────────────────────
    st.markdown('<p class="section-title">Categorical Feature Analysis</p>',
                unsafe_allow_html=True)

    available_cat = [c for c in CAT_FEATS if c in df.columns]
    chosen_cat = st.selectbox("Select categorical feature", available_cat, key="cat_sel")

    ct = pd.crosstab(df[chosen_cat], df[TARGET], normalize="index") * 100
    # Ensure both columns exist even if one class is absent
    for cls in ["no", "yes"]:
        if cls not in ct.columns:
            ct[cls] = 0.0
    ct = ct[["no", "yes"]]

    fig, ax = plt.subplots(figsize=(12, 4))
    ct.plot(kind="bar", ax=ax,
            color=["#f97316", "#2563eb"],
            edgecolor="white", width=0.7)
    ax.set_title(f"Subscription Rate by {chosen_cat} (%)", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Subscribed")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Correlation heatmap ────────────────────────────────────
    st.markdown('<p class="section-title">Correlation Heatmap (Numeric Features)</p>',
                unsafe_allow_html=True)

    df_num = df[available_num].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Numeric Feature Correlation Matrix", fontweight="bold")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Month-wise subscription rate ───────────────────────────
    st.markdown('<p class="section-title">Subscription Rate by Month</p>',
                unsafe_allow_html=True)

    month_order = ["jan","feb","mar","apr","may","jun",
                   "jul","aug","sep","oct","nov","dec"]
    ct_month = (
        pd.crosstab(df["MONTH"], df[TARGET], normalize="index")["yes"] * 100
    ).reindex([m for m in month_order if m in df["MONTH"].unique()])

    fig, ax = plt.subplots(figsize=(11, 4))
    ct_month.plot(kind="bar", ax=ax, color="#2563eb", edgecolor="white", width=0.7)
    ax.set_title("% Subscribed by Contact Month", fontweight="bold")
    ax.set_ylabel("Subscription Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Model Training</p>', unsafe_allow_html=True)

    if not selected_features:
        st.warning("⚠️ Please select at least one independent variable from the sidebar.")
        st.stop()

    # Preprocess
    X, y, scaler, encoders, le_y = preprocess(df, selected_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📌 Configuration")
        st.write(f"- **Algorithm:** {model_name}")
        st.write(f"- **Target Variable:** `{TARGET}` — Term Deposit Subscription")
        st.write(f"- **Features selected ({len(selected_features)}):** {', '.join(selected_features)}")
        st.write(f"- **Train samples:** {len(X_train):,}")
        st.write(f"- **Test samples:** {len(X_test):,}")
        st.write(f"- **Class balance (train):** "
                 f"Yes={int(y_train.sum())} | No={int((y_train==0).sum())}")

    with col2:
        st.markdown("#### 🚀 Train")
        if st.button("Train Model", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_name} …"):
                mdl = MODELS[model_name]
                mdl.fit(X_train, y_train)
                # Save everything needed later
                st.session_state.update({
                    "model":    mdl,
                    "X_train":  X_train, "y_train": y_train,
                    "X_test":   X_test,  "y_test":  y_test,
                    "scaler":   scaler,
                    "encoders": encoders,
                    "le_y":     le_y,
                    "feats":    selected_features,
                    "mdl_name": model_name,
                })
                st.success(f"✅ {model_name} trained successfully!")

    if "model" in st.session_state:
        mdl    = st.session_state["model"]
        X_tr   = st.session_state["X_train"]
        y_tr   = st.session_state["y_train"]
        X_te   = st.session_state["X_test"]
        y_te   = st.session_state["y_test"]

        acc_tr = accuracy_score(y_tr, mdl.predict(X_tr))
        acc_te = accuracy_score(y_te, mdl.predict(X_te))

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Train Accuracy", f"{acc_tr:.2%}")
        m2.metric("🧪 Test Accuracy",  f"{acc_te:.2%}")
        m3.metric("📉 Overfitting Gap", f"{abs(acc_tr - acc_te):.2%}")

        # Feature importance (tree-based only)
        if hasattr(mdl, "feature_importances_"):
            st.markdown('<p class="section-title">Feature Importance</p>',
                        unsafe_allow_html=True)
            fi = pd.Series(mdl.feature_importances_,
                           index=st.session_state["feats"]).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(9, max(4, len(fi) * 0.4)))
            fi.plot(kind="barh", ax=ax, color="#2563eb", edgecolor="white")
            ax.set_title("Feature Importance", fontweight="bold")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)
            plt.close()

        # Logistic Regression coefficients
        if hasattr(mdl, "coef_"):
            st.markdown('<p class="section-title">Feature Coefficients (Logistic Regression)</p>',
                        unsafe_allow_html=True)
            coef_df = pd.DataFrame({
                "Feature":     st.session_state["feats"],
                "Coefficient": mdl.coef_[0],
            }).sort_values("Coefficient", ascending=True)

            fig, ax = plt.subplots(figsize=(9, max(4, len(coef_df) * 0.4)))
            colors = ["#f97316" if v < 0 else "#2563eb"
                      for v in coef_df["Coefficient"]]
            ax.barh(coef_df["Feature"], coef_df["Coefficient"],
                    color=colors, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Logistic Regression Coefficients\n"
                         "(Blue = positive → more likely to subscribe | "
                         "Orange = negative)", fontweight="bold", fontsize=10)
            ax.set_xlabel("Coefficient Value")
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Model Evaluation</p>', unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.info("👈 Go to **Model Training** tab and train a model first.")
    else:
        mdl    = st.session_state["model"]
        X_te   = st.session_state["X_test"]
        y_te   = st.session_state["y_test"]   # integer labels (0/1)
        y_pred = mdl.predict(X_te)

        col1, col2 = st.columns(2)

        # ── Confusion Matrix ──────────────────────────────────
        with col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_te, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            # FIX: display_labels explicitly set so it shows No/Yes, not 0/1
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=["No", "Yes"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("Confusion Matrix", fontweight="bold")
            st.pyplot(fig)
            plt.close()

        # ── ROC Curve ─────────────────────────────────────────
        with col2:
            st.markdown("#### ROC Curve")
            if hasattr(mdl, "predict_proba"):
                # FIX: y_te is already int (0/1) — no LabelEncoder needed
                y_prob = mdl.predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, y_prob)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, color="#2563eb", lw=2,
                        label=f"AUC = {roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], color="gray",
                        linestyle="--", lw=1, label="Random Classifier")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve", fontweight="bold")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close()
            else:
                st.info("ROC curve requires predict_proba — not available for this model.")

        st.divider()

        # ── Key Metrics ────────────────────────────────────────
        st.markdown('<p class="section-title">Performance Metrics</p>',
                    unsafe_allow_html=True)

        cm_flat = confusion_matrix(y_te, y_pred).ravel()
        tn, fp, fn, tp = cm_flat
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy",  f"{accuracy_score(y_te, y_pred):.2%}")
        k2.metric("Precision", f"{precision:.2%}")
        k3.metric("Recall",    f"{recall:.2%}")
        k4.metric("F1-Score",  f"{f1:.2%}")

        # ── Classification Report ──────────────────────────────
        st.markdown('<p class="section-title">Full Classification Report</p>',
                    unsafe_allow_html=True)
        report = classification_report(y_te, y_pred,
                                        target_names=["No", "Yes"],
                                        output_dict=True)
        report_df = pd.DataFrame(report).T

        # FIX: only gradient on numeric columns that actually exist
        numeric_report_cols = [c for c in ["precision","recall","f1-score"]
                                if c in report_df.columns]
        styled = (report_df
                  .style
                  .format("{:.3f}", subset=numeric_report_cols)
                  .background_gradient(cmap="Blues", subset=numeric_report_cols))
        st.dataframe(styled, use_container_width=True)

        st.divider()

        # ── Business Impact: Call Reduction ────────────────────
        st.markdown('<p class="section-title">📞 Business Impact: Call Reduction Analysis</p>',
                    unsafe_allow_html=True)

        if hasattr(mdl, "predict_proba"):
            y_prob_all = mdl.predict_proba(X_te)[:, 1]
            threshold  = st.slider("Exclusion threshold (exclude if prob < X%)",
                                   1, 20, 5, key="thresh") / 100

            excluded        = y_prob_all < threshold
            calls_saved     = int(excluded.sum())
            missed_subs     = int((excluded & (y_te == 1)).sum())
            total_subs      = int((y_te == 1).sum())
            total_customers = len(y_te)

            cost_saved = calls_saved * 5
            opp_loss   = missed_subs * 200
            net_profit = cost_saved - opp_loss

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Calls Saved",       f"{calls_saved:,}",
                      f"{calls_saved/total_customers:.1%} of test set")
            b2.metric("Subscribers Missed",f"{missed_subs}",
                      f"{missed_subs/total_subs:.1%} of subscribers")
            b3.metric("Cost Saved (€5/call)",  f"€{cost_saved:,}")
            b4.metric("Net Profit Impact",     f"€{net_profit:,}")

            # Probability distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(y_prob_all[y_te == 0], bins=40, alpha=0.6,
                    color="#f97316", label="Actual: No")
            ax.hist(y_prob_all[y_te == 1], bins=40, alpha=0.6,
                    color="#2563eb", label="Actual: Yes")
            ax.axvline(threshold, color="red", linestyle="--",
                       label=f"Exclusion Threshold ({threshold:.0%})")
            ax.set_title("Predicted Probability Distribution by Actual Class",
                         fontweight="bold")
            ax.set_xlabel("Predicted Probability of Subscribing")
            ax.set_ylabel("Number of Customers")
            ax.legend()
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════════════════════════════
# TAB 5 — PREDICT NEW CLIENT
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Predict for a New Client</p>',
                unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.info("👈 Go to **Model Training** tab and train a model first.")
    else:
        mdl      = st.session_state["model"]
        feats    = st.session_state["feats"]
        scaler   = st.session_state["scaler"]
        encoders = st.session_state["encoders"]
        le_y     = st.session_state["le_y"]

        st.markdown("Enter the client details and click **Predict** below.")

        input_data = {}
        cols_ui = st.columns(3)
        for i, feat in enumerate(feats):
            with cols_ui[i % 3]:
                if feat in CAT_FEATS:
                    input_data[feat] = st.selectbox(
                        feat, CAT_OPTIONS[feat], key=f"pred_{feat}"
                    )
                else:
                    cfg = NUM_CFG.get(feat, {"min": 0, "max": 10000, "default": 0})
                    input_data[feat] = st.number_input(
                        feat,
                        min_value=cfg["min"],
                        max_value=cfg["max"],
                        value=cfg["default"],
                        key=f"pred_{feat}"
                    )

        st.markdown("---")
        if st.button("🔮 Predict Subscription", type="primary",
                     use_container_width=True):
            # Build input row
            input_df = pd.DataFrame([input_data])

            # FIX: use the SAME encoders fitted during training
            for col in feats:
                if col in CAT_FEATS and col in encoders:
                    le = encoders[col]
                    val = input_df[col].astype(str).iloc[0]
                    # Handle unseen labels gracefully
                    if val in le.classes_:
                        input_df[col] = le.transform([val])[0]
                    else:
                        input_df[col] = 0

            input_df = input_df[feats].astype(float)
            input_scaled = pd.DataFrame(
                scaler.transform(input_df), columns=feats
            )

            # Predict
            pred_int = int(mdl.predict(input_scaled)[0])
            # FIX: decode integer back to "yes"/"no" using le_y
            pred_label = le_y.inverse_transform([pred_int])[0]  # "no" or "yes"

            if pred_label == "yes":
                st.success(
                    "✅ **Prediction: YES** — This client is likely to subscribe to the term deposit!"
                )
            else:
                st.error(
                    "❌ **Prediction: NO** — This client is unlikely to subscribe to the term deposit."
                )

            # Show probabilities if available
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(input_scaled)[0]
                # classes_ is [0, 1]; map to no/yes via le_y
                class_labels = le_y.inverse_transform(mdl.classes_)
                prob_dict = dict(zip(class_labels, proba))

                p1, p2 = st.columns(2)
                p1.metric("Probability — YES",
                          f"{prob_dict.get('yes', 0):.2%}")
                p2.metric("Probability — NO",
                          f"{prob_dict.get('no', 0):.2%}")

                # Gauge bar
                yes_prob = prob_dict.get("yes", 0)
                fig, ax = plt.subplots(figsize=(8, 1.5))
                ax.barh([""], [yes_prob], color="#2563eb", height=0.5)
                ax.barh([""], [1 - yes_prob], left=[yes_prob],
                        color="#f97316", height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Subscription Probability")
                ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
                ax.set_title(f"Probability Gauge  (Yes: {yes_prob:.1%} | No: {1-yes_prob:.1%})",
                             fontweight="bold")
                st.pyplot(fig)
                plt.close()
