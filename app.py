import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Probability of Default",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_package = jb.load("mLogReg.pkl")

    # Model was saved as:
    # {
    #     "model": final_model,
    #     "threshold": 0.30
    # }

    model = model_package["model"]
    threshold = model_package["threshold"]

    return model, threshold


model, threshold = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("💳 Probability of Default")

st.markdown(
    """
    Estimate the **Probability of Default (PD)** for a loan applicant
    using the trained and calibrated Logistic Regression model.
    """
)

st.divider()


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.header("📋 Loan Applicant Information")

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

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

    term = st.selectbox(
        "Loan Term",
        [
            "36 months",
            "60 months"
        ]
    )

    emp_length_int = st.number_input(
        "Employment Length (years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )

    mths_since_issue_d = st.number_input(
        "Months Since Loan Issue",
        min_value=36,
        max_value=124,
        value=60,
        step=1
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=5.42,
        max_value=26.06,
        value=12.00,
        step=0.01
    )

    mths_since_earliest_cr_line = st.number_input(
        "Months Since Earliest Credit Line",
        min_value=77,
        max_value=587,
        value=180,
        step=1
    )

    acc_now_delinq = st.number_input(
        "Current Delinquencies",
        min_value=0,
        max_value=2,
        value=0,
        step=1
    )

    inq_last_6mths = st.number_input(
        "Inquiries in Last 6 Months",
        min_value=0,
        max_value=8,
        value=0,
        step=1
    )

    annual_inc = st.number_input(
        "Annual Income ($)",
        min_value=9696.0,
        max_value=1362000.0,
        value=60000.0,
        step=1000.0
    )

    dti = st.number_input(
        "Debt-to-Income Ratio (%)",
        min_value=0.0,
        max_value=39.81,
        value=15.0,
        step=0.01
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

term_number = float(
    term.replace(" months", "")
)

rate_term = int_rate * term_number

log_annual_inc = np.log1p(
    annual_inc
)

credit_history_years = (
    mths_since_earliest_cr_line / 12
)

has_recent_inquiry = int(
    inq_last_6mths > 0
)

has_delinquency = int(
    acc_now_delinq > 0
)

long_employment = int(
    emp_length_int >= 5
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Calculate Probability of Default",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # =====================================================
    # CREATE MODEL INPUT
    # =====================================================

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
        ],

        "rate_term": [
            rate_term
        ],

        "log_annual_inc": [
            log_annual_inc
        ],

        "credit_history_years": [
            credit_history_years
        ],

        "has_recent_inquiry": [
            has_recent_inquiry
        ],

        "has_delinquency": [
            has_delinquency
        ],

        "long_employment": [
            long_employment
        ]
    })


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    probability_good = model.predict_proba(
        input_data
    )[0][1]

    probability_default = (
        1 - probability_good
    )


    # =====================================================
    # DEFAULT CLASSIFICATION
    # =====================================================

    if probability_default >= threshold:

        default_prediction = "Default Risk"

    else:

        default_prediction = "Non-Default"


    # =====================================================
    # RISK CATEGORY
    # =====================================================

    if probability_default < 0.05:

        risk = "Low"

    elif probability_default < 0.10:

        risk = "Moderate"

    elif probability_default < 0.20:

        risk = "High"

    else:

        risk = "Very High"


    # =====================================================
    # MODEL EXPLANATION
    # =====================================================

    st.divider()

    st.subheader("🔎 Model Explanation")

    st.write(
        """
        The explanation below shows the features that have the
        strongest contribution to the underlying Logistic Regression
        prediction.

        Positive values increase the model's predicted default risk,
        while negative values decrease it.
        """
    )


    # -----------------------------------------------------
    # GET UNDERLYING LOGISTIC REGRESSION MODEL
    # -----------------------------------------------------

    try:

        # CalibratedClassifierCV contains several fitted
        # Logistic Regression models.

        calibrated_classifier = (
            model.calibrated_classifiers_[0]
        )

        base_estimator = (
            calibrated_classifier.estimator
        )

        preprocessor = (
            base_estimator.named_steps["preprocessor"]
        )

        lr_model = (
            base_estimator.named_steps["model"]
        )


        # -------------------------------------------------
        # TRANSFORM INPUT DATA
        # -------------------------------------------------

        transformed_data = (
            preprocessor.transform(
                input_data
            )
        )


        # Convert sparse matrix to dense
        if hasattr(
            transformed_data,
            "toarray"
        ):

            transformed_data = (
                transformed_data.toarray()
            )


        # -------------------------------------------------
        # FEATURE NAMES
        # -------------------------------------------------

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        # -------------------------------------------------
        # LOGISTIC REGRESSION COEFFICIENTS
        # -------------------------------------------------

        coefficients = (
            lr_model.coef_[0]
        )


        # -------------------------------------------------
        # FEATURE CONTRIBUTION
        # -------------------------------------------------

        contributions = (
            transformed_data[0]
            * coefficients
        )


        explanation_df = pd.DataFrame({

            "Feature": feature_names,

            "Contribution": contributions

        })


        explanation_df[
            "Absolute Impact"
        ] = (
            explanation_df[
                "Contribution"
            ].abs()
        )


        # -------------------------------------------------
        # TOP 10 FEATURES
        # -------------------------------------------------

        explanation_df = (
            explanation_df
            .sort_values(
                "Absolute Impact",
                ascending=False
            )
            .head(10)
        )


        # -------------------------------------------------
        # DISPLAY CHART
        # -------------------------------------------------

        st.bar_chart(

            explanation_df
            .set_index("Feature")[
                "Contribution"
            ]

        )


    except Exception as e:

        st.warning(
            "Feature contribution explanation "
            "could not be generated."
        )

        st.caption(
            f"Explanation error: {e}"
        )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.header("📊 Prediction Result")


    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # PD
    # -----------------------------------------------------

    with col1:

        st.metric(
            "Probability of Default",
            f"{probability_default:.2%}"
        )


    # -----------------------------------------------------
    # NON-DEFAULT PROBABILITY
    # -----------------------------------------------------

    with col2:

        st.metric(
            "Probability of Non-Default",
            f"{probability_good:.2%}"
        )


    # -----------------------------------------------------
    # DEFAULT DECISION
    # -----------------------------------------------------

    with col3:

        st.metric(
            "Default Decision",
            default_prediction
        )


    # =====================================================
    # RISK CATEGORY
    # =====================================================

    st.subheader("Risk Category")

    st.write(
        f"### {risk}"
    )


    # =====================================================
    # DEFAULT PROBABILITY BAR
    # =====================================================

    st.subheader(
        "Default Probability"
    )

    st.progress(
        float(probability_default),
        text=(
            f"Estimated PD: "
            f"{probability_default:.2%}"
        )
    )


    st.caption(
        "PD represents the model-estimated "
        "probability of default."
    )


    st.caption(
        f"Default classification threshold: "
        f"{threshold:.0%}"
    )


    # =====================================================
    # APPLICANT SUMMARY
    # =====================================================

    st.divider()

    st.subheader(
        "Applicant Summary"
    )


    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # COLUMN 1
    # -----------------------------------------------------

    with col1:

        st.write(
            "**Grade:**",
            grade
        )

        st.write(
            "**Home Ownership:**",
            home_ownership
        )

        st.write(
            "**Loan Purpose:**",
            purpose
        )


    # -----------------------------------------------------
    # COLUMN 2
    # -----------------------------------------------------

    with col2:

        st.write(
            "**Loan Term:**",
            term
        )

        st.write(
            "**Interest Rate:**",
            f"{int_rate:.2f}%"
        )

        st.write(
            "**Annual Income:**",
            f"${annual_inc:,.0f}"
        )


    # -----------------------------------------------------
    # COLUMN 3
    # -----------------------------------------------------

    with col3:

        st.write(
            "**DTI:**",
            f"{dti:.2f}%"
        )

        st.write(
            "**Employment:**",
            f"{emp_length_int} years"
        )

        st.write(
            "**Credit History:**",
            f"{credit_history_years:.1f} years"
        )


    # =====================================================
    # MODEL INPUT DATA
    # =====================================================

    st.divider()

    with st.expander(
        "View Model Input Data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )