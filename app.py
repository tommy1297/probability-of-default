import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Probability of Default",
    page_icon="💳",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_package = jb.load("mLogReg.pkl")

    model = model_package["model"]
    threshold = model_package["threshold"]

    return model, threshold


model, threshold = load_model()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    df = df.copy()

    # Loan term is already expressed in months
    term_months = df["term"].astype(float)

    # Feature 1: Interest Rate × Term
    df["rate_term"] = (
        df["int_rate"] * term_months
    )

    # Feature 2: Log Annual Income
    df["log_annual_inc"] = (
        np.log1p(df["annual_inc"])
    )

    # Feature 3: Credit History in Years
    df["credit_history_years"] = (
        df["mths_since_earliest_cr_line"] / 12
    )

    # Feature 4: Recent Inquiry Indicator
    df["has_recent_inquiry"] = (
        df["inq_last_6mths"] > 0
    ).astype(int)

    # Feature 5: Delinquency Indicator
    df["has_delinquency"] = (
        df["acc_now_delinq"].fillna(0) > 0
    ).astype(int)

    # Feature 6: Long Employment Indicator
    df["long_employment"] = (
        df["emp_length_int"] >= 5
    ).astype(int)

    return df


# ============================================================
# TITLE
# ============================================================

st.title("💳 Probability of Default")

st.write(
    "Estimate the Probability of Default (PD) for a loan "
    "applicant using the trained calibrated Logistic "
    "Regression model."
)


# ============================================================
# LOAN APPLICANT INFORMATION
# ============================================================

st.header("📋 Loan Applicant Information")


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "MORTGAGE",
            "OWN"
        ]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "debt_consolidation",
            "major_purchase",
            "home_improvement",
            "credit_card",
            "medical",
            "vacation",
            "other",
            "moving",
            "small_business",
            "house",
            "wedding",
            "car",
            "educational",
            "renewable_energy"
        ]
    )

    verification_status = st.selectbox(
        "Verification Status",
        [
            "Verified",
            "Source Verified",
            "Not Verified"
        ]
    )

    # Loan term is entered in months
    # Step = 12 means 12, 24, 36, 48, 60, ...
    term = st.number_input(
        "Loan Term (months)",
        min_value=0,
        value=36,
        step=12
    )

    emp_length_int = st.number_input(
        "Employment Length (years)",
        min_value=0,
        value=5,
        step=1
    )

    mths_since_issue_d = st.number_input(
        "Months Since Loan Issue",
        min_value=0,
        value=60,
        step=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=12.00,
        step=0.01
    )

    mths_since_earliest_cr_line = st.number_input(
        "Months Since Earliest Credit Line",
        min_value=0,
        value=180,
        step=1
    )

    acc_now_delinq = st.number_input(
        "Current Delinquencies",
        min_value=0,
        value=0,
        step=1
    )

    inq_last_6mths = st.number_input(
        "Inquiries in Last 6 Months",
        min_value=0,
        value=0,
        step=1
    )

    annual_inc = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=60000.0,
        step=1000.0
    )

    dti = st.number_input(
        "Debt-to-Income Ratio (%)",
        min_value=0.0,
        value=15.0,
        step=0.01
    )


# ============================================================
# INPUT RANGE NOTE
# ============================================================

st.caption(
    "Note: Predictions for inputs outside the range observed "
    "in the training data may be less reliable."
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Calculate Probability of Default",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE APPLICANT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "grade": [grade],

        "home_ownership": [
            home_ownership
        ],

        "purpose": [
            purpose
        ],

        "verification_status": [
            verification_status
        ],

        "term": [
            term
        ],

        "emp_length_int": [
            emp_length_int
        ],

        "mths_since_issue_d": [
            mths_since_issue_d
        ],

        "int_rate": [
            int_rate
        ],

        "mths_since_earliest_cr_line": [
            mths_since_earliest_cr_line
        ],

        "acc_now_delinq": [
            acc_now_delinq
        ],

        "inq_last_6mths": [
            inq_last_6mths
        ],

        "annual_inc": [
            annual_inc
        ],

        "dti": [
            dti
        ]
    })


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    input_data = feature_engineering(
        input_data
    )


    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    model_features = [

        "grade",

        "home_ownership",

        "purpose",

        "verification_status",

        "term",

        "emp_length_int",

        "mths_since_issue_d",

        "int_rate",

        "mths_since_earliest_cr_line",

        "acc_now_delinq",

        "inq_last_6mths",

        "annual_inc",

        "dti",

        "rate_term",

        "log_annual_inc",

        "credit_history_years",

        "has_recent_inquiry",

        "has_delinquency",

        "long_employment"
    ]


    input_data = input_data[
        model_features
    ]


    # --------------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------------

    prob_good = model.predict_proba(
        input_data
    )[:, 1]


    # Good_Bad = 1 means non-default
    # Therefore:
    # Probability of Default = 1 - Probability of Good

    prob_default = (
        1 - prob_good
    )


    prob_default = float(
        prob_default[0]
    )


    prob_non_default = (
        1 - prob_default
    )


    # --------------------------------------------------------
    # APPLY CLASSIFICATION THRESHOLD
    # --------------------------------------------------------

    default_prediction = (
        prob_default >= threshold
    )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.header("📊 Prediction Result")


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Probability of Default",
            f"{prob_default:.2%}"
        )


    with col2:

        st.metric(
            "Probability of Non-Default",
            f"{prob_non_default:.2%}"
        )


    # --------------------------------------------------------
    # DEFAULT PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "Default Prediction"
    )


    if default_prediction:

        st.error(
            f"**Default**\n\n"
            f"PD ({prob_default:.2%}) is above "
            f"the selected threshold "
            f"({threshold:.0%})."
        )

    else:

        st.success(
            f"**Non-Default**\n\n"
            f"PD ({prob_default:.2%}) is below "
            f"the selected threshold "
            f"({threshold:.0%})."
        )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.info(
        f"""
        **How to interpret the result**

        The model estimates a **{prob_default:.2%}**
        probability of default for this applicant.

        The classification threshold used in this project
        is **{threshold:.0%}**.

        Therefore:

        - PD ≥ {threshold:.0%} → Default
        - PD < {threshold:.0%} → Non-Default
        """
    )


    # ========================================================
    # APPLICANT SUMMARY
    # ========================================================

    st.subheader(
        "👤 Applicant Summary"
    )


    summary_col1, summary_col2 = st.columns(2)


    with summary_col1:

        st.write(
            f"**Grade:** {grade}"
        )

        st.write(
            f"**Home Ownership:** "
            f"{home_ownership}"
        )

        st.write(
            f"**Loan Purpose:** "
            f"{purpose}"
        )

        st.write(
            f"**Loan Term:** "
            f"{term:.0f} months"
        )

        st.write(
            f"**Interest Rate:** "
            f"{int_rate:.2f}%"
        )

        st.write(
            f"**Annual Income:** "
            f"{annual_inc:,.0f}"
        )


    with summary_col2:

        st.write(
            f"**DTI:** "
            f"{dti:.2f}%"
        )

        st.write(
            f"**Employment:** "
            f"{emp_length_int:.0f} years"
        )

        st.write(
            f"**Credit History:** "
            f"{mths_since_earliest_cr_line / 12:.1f} years"
        )

        st.write(
            f"**Current Delinquencies:** "
            f"{acc_now_delinq:.0f}"
        )

        st.write(
            f"**Recent Inquiries:** "
            f"{inq_last_6mths:.0f}"
        )


    # ========================================================
    # MODEL INPUT DATA
    # ========================================================

    with st.expander(
        "View Model Input Data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )