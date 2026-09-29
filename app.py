import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HAB Decision Support System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Harmful Algal Bloom Decision Support System")

st.info(
    "This system estimates proxy-derived HAB risk using KSPCB "
    "water-quality data. Results do not represent confirmed algal blooms."
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # Load XGBoost model using native JSON format
    # --------------------------------------------------------

    model = xgb.XGBClassifier()

    model.load_model(
        "xgb_binary_hab_model.json"
    )

    # --------------------------------------------------------
    # Re-create SHAP explainer locally
    # Avoids Colab/Windows pickle compatibility problems
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    # --------------------------------------------------------
    # Load feature names
    # --------------------------------------------------------

    features = joblib.load(
        "hab_feature_names.pkl"
    )

    # --------------------------------------------------------
    # Load HAB risk thresholds
    # --------------------------------------------------------

    thresholds = joblib.load(
        "hab_risk_thresholds.pkl"
    )

    return model, explainer, features, thresholds


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "KSPCB_HAB_Final_Dataset.csv"
    )

    reference = pd.read_csv(
        "HAB_Proxy_Reference_Distributions.csv"
    )

    feature_ranges = pd.read_csv(
        "HAB_Feature_Ranges.csv"
    )

    forecast = pd.read_csv(
        "Rayasandra_ARIMA_Forecast.csv"
    )

    return data, reference, feature_ranges, forecast


model, explainer, features, thresholds = load_models()

data, reference, feature_ranges, forecast = load_data()


# ============================================================
# THRESHOLDS
# ============================================================

q1 = thresholds["q1"]
q2 = thresholds["q2"]
q3 = thresholds["q3"]


# ============================================================
# FUNCTIONS
# ============================================================

def percentile_score(series, value):

    series = series.dropna().values

    return (
        np.sum(series <= value)
        / len(series)
        * 100
    )


def calculate_proxy_score(values):

    # --------------------------------------------------------
    # Nutrient Risk
    # --------------------------------------------------------

    nitrate_risk = percentile_score(
        reference["Nitrate"],
        values["Nitrate"]
    )

    ammonia_risk = percentile_score(
        reference["Ammonical_N"],
        values["Ammonical_N"]
    )

    phosphate_risk = percentile_score(
        reference["Phosphate"],
        values["Phosphate"]
    )

    # --------------------------------------------------------
    # Organic Pollution Risk
    # --------------------------------------------------------

    bod_risk = percentile_score(
        reference["BOD"],
        values["BOD"]
    )

    cod_risk = percentile_score(
        reference["COD"],
        values["COD"]
    )

    # --------------------------------------------------------
    # Ecological Risk
    # --------------------------------------------------------

    turbidity_risk = percentile_score(
        reference["Turbidity"],
        values["Turbidity"]
    )

    do_percentile = percentile_score(
        reference["DO"],
        values["DO"]
    )

    # Lower DO represents higher ecological stress
    do_risk = 100 - do_percentile

    # pH deviation from reference point
    ph_reference = abs(
        reference["pH"] - 7.5
    )

    ph_value = abs(
        values["pH"] - 7.5
    )

    ph_risk = percentile_score(
        ph_reference,
        ph_value
    )

    # --------------------------------------------------------
    # Component Scores
    # --------------------------------------------------------

    nutrient_risk = np.mean([
        nitrate_risk,
        ammonia_risk,
        phosphate_risk
    ])

    organic_risk = np.mean([
        bod_risk,
        cod_risk
    ])

    ecological_risk = np.mean([
        do_risk,
        ph_risk,
        turbidity_risk
    ])

    # Equal conceptual component weighting
    final_score = np.mean([
        nutrient_risk,
        organic_risk,
        ecological_risk
    ])

    return np.clip(
        final_score,
        0,
        100
    )


def get_risk_class(score):

    if score <= q1:

        return "Low"

    elif score <= q2:

        return "Moderate"

    elif score <= q3:

        return "High"

    else:

        return "Very High"


def get_dss_recommendation(risk_class):

    if risk_class == "Low":

        return (
            "Routine Monitoring",
            "Continue regular water-quality monitoring."
        )

    elif risk_class == "Moderate":

        return (
            "Enhanced Monitoring",
            "Increase monitoring frequency and observe nutrient "
            "and organic pollution indicators."
        )

    elif risk_class == "High":

        return (
            "Field Investigation",
            "Conduct field inspection and investigate possible "
            "nutrient and sewage-related pollution."
        )

    else:

        return (
            "Early Warning",
            "Prioritize field and laboratory verification, "
            "inspect pollution inflows, and assess appropriate "
            "mitigation measures."
        )


def check_ranges(values):

    warnings = []

    for feature in features:

        row = feature_ranges[
            feature_ranges["Feature"] == feature
        ]

        if row.empty:
            continue

        minimum = float(
            row["Minimum"].iloc[0]
        )

        maximum = float(
            row["Maximum"].iloc[0]
        )

        if (
            values[feature] < minimum
            or values[feature] > maximum
        ):

            warnings.append(
                f"{feature}: entered value {values[feature]} "
                f"is outside historical range "
                f"{minimum:.2f}–{maximum:.2f}"
            )

    return warnings


def show_dss_box(risk_class):

    action, recommendation = (
        get_dss_recommendation(
            risk_class
        )
    )

    if risk_class == "Very High":

        st.error(
            f"🔴 {action}"
        )

    elif risk_class == "High":

        st.warning(
            f"🟠 {action}"
        )

    elif risk_class == "Moderate":

        st.info(
            f"🟡 {action}"
        )

    else:

        st.success(
            f"🟢 {action}"
        )

    st.write(
        recommendation
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "🏞️ Lake Monitoring",
        "🧪 New Sample Prediction"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.header(
        "HAB Risk Decision Support"
    )

    st.write(
        """
        This dashboard integrates water-quality analysis,
        machine learning, explainable AI, forecasting,
        and decision-support recommendations.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Monitoring Records",
            len(data)
        )

    with col2:

        st.metric(
            "Monitoring Stations",
            int(
                data["STN_Code"].nunique()
            )
        )

    with col3:

        st.metric(
            "ML Features",
            len(features)
        )

    st.subheader(
        "System Workflow"
    )

    st.write(
        """
        **KSPCB Data**
        → Data Processing
        → HAB Risk Proxy
        → XGBoost Early Warning
        → SHAP Explanation
        → ARIMA Forecasting
        → DSS Recommendation
        """
    )

    st.subheader(
        "About the System"
    )

    st.write(
        """
        The system uses eight water-quality indicators:

        **DO, pH, BOD, COD, Nitrate, Ammonical-N,
        Phosphate and Turbidity.**

        The HAB Risk Score is a proxy-derived environmental
        risk indicator. It does not indicate laboratory-confirmed
        harmful algal bloom occurrence.
        """
    )


# ============================================================
# LAKE MONITORING
# ============================================================

elif page == "🏞️ Lake Monitoring":

    st.header(
        "🏞️ Lake Monitoring Dashboard"
    )

    # --------------------------------------------------------
    # Prepare station data
    # --------------------------------------------------------

    station_data = (
        data
        .dropna(
            subset=[
                "STN_Code",
                "Lake_Name"
            ]
        )
        .copy()
    )

    station_data["Station_Label"] = (
        station_data["STN_Code"]
        .astype(int)
        .astype(str)
        + " - "
        + station_data["Lake_Name"]
        .astype(str)
    )

    station_options = sorted(
        station_data[
            "Station_Label"
        ].unique()
    )

    # --------------------------------------------------------
    # Station selector
    # --------------------------------------------------------

    selected_label = st.selectbox(
        "Select Monitoring Station",
        station_options
    )

    selected_code = int(
        selected_label
        .split(" - ")[0]
    )

    selected_data = (
        station_data[
            station_data["STN_Code"]
            == selected_code
        ]
        .copy()
    )

    selected_data[
        "Report_Date"
    ] = pd.to_datetime(
        selected_data[
            "Report_Date"
        ]
    )

    selected_data = (
        selected_data
        .sort_values(
            "Report_Date"
        )
    )

    latest = (
        selected_data
        .iloc[-1]
    )


    # ========================================================
    # CURRENT STATUS
    # ========================================================

    st.subheader(
        "Latest Monitoring Status"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "HAB Proxy Score",
            f"{latest['HAB_Risk_Score']:.2f}/100"
        )

    with c2:

        st.metric(
            "Risk Class",
            latest[
                "HAB_Risk_Class"
            ]
        )

    with c3:

        st.metric(
            "Latest Monitoring",
            latest[
                "Report_Date"
            ].strftime("%b %Y")
        )


    # ========================================================
    # HISTORICAL RISK
    # ========================================================

    st.subheader(
        "Historical HAB Risk"
    )

    risk_chart = (
        selected_data[
            [
                "Report_Date",
                "HAB_Risk_Score"
            ]
        ]
        .set_index(
            "Report_Date"
        )
    )

    st.line_chart(
        risk_chart
    )


    # ========================================================
    # LATEST WATER QUALITY
    # ========================================================

    st.subheader(
        "Latest Water Quality"
    )

    latest_water_quality = (
        latest[
            features
        ]
        .astype(float)
        .to_frame(
            name="Value"
        )
    )

    st.dataframe(
        latest_water_quality,
        use_container_width=True
    )


    # ========================================================
    # ML EARLY WARNING
    # ========================================================

    st.subheader(
        "🤖 Next-Month ML Early Warning"
    )

    latest_features = (
        latest[
            features
        ]
        .astype(float)
        .to_frame()
        .T
    )

    latest_features = (
        latest_features[
            features
        ]
    )

    prediction = (
        model.predict(
            latest_features
        )[0]
    )

    probability = (
        model.predict_proba(
            latest_features
        )[0, 1]
    )

    prediction_label = (
        "Elevated Risk"
        if prediction == 1
        else "Lower Risk"
    )

    m1, m2 = st.columns(2)

    with m1:

        st.metric(
            "Next-Month Prediction",
            prediction_label
        )

    with m2:

        st.metric(
            "Elevated-Risk Probability",
            f"{probability * 100:.2f}%"
        )

    if prediction == 1:

        st.warning(
            "⚠️ The machine-learning model indicates elevated "
            "proxy-risk conditions for the next month."
        )

    else:

        st.success(
            "The machine-learning model indicates lower "
            "proxy-risk conditions for the next month."
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.subheader(
        "🔍 Main Model Drivers"
    )

    shap_values = explainer.shap_values(
        latest_features
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        shap_values_current = (
            shap_values[-1][0]
        )

    else:

        shap_values_current = (
            np.array(shap_values)[0]
        )

    driver_table = pd.DataFrame({

        "Feature":
            features,

        "Current Value":
            latest_features
            .iloc[0]
            .values,

        "SHAP Value":
            shap_values_current
    })

    driver_table[
        "Absolute SHAP"
    ] = (
        driver_table[
            "SHAP Value"
        ].abs()
    )

    driver_table = (
        driver_table
        .sort_values(
            "Absolute SHAP",
            ascending=False
        )
    )

    st.dataframe(
        driver_table[
            [
                "Feature",
                "Current Value",
                "SHAP Value"
            ]
        ]
        .head(5)
        .round(3),

        use_container_width=True
    )

    st.caption(
        "Positive SHAP values generally push the prediction "
        "toward Elevated Risk, while negative SHAP values "
        "generally push it toward Lower Risk."
    )

    st.caption(
        "SHAP values describe model influence and should not "
        "be interpreted as proof of environmental causality."
    )


    # ========================================================
    # DSS RECOMMENDATION
    # ========================================================

    st.subheader(
        "📋 DSS Recommendation"
    )

    show_dss_box(
        latest[
            "HAB_Risk_Class"
        ]
    )

    st.caption(
        "Recommendations are decision-support guidance and "
        "do not replace field sampling, laboratory verification, "
        "or regulatory assessment."
    )


    # ========================================================
    # RAYASANDRA FORECAST
    # ========================================================

    if selected_code == 4531:

        st.divider()

        st.subheader(
            "📈 Rayasandra Lake ARIMA Forecast"
        )

        st.caption(
            "ARIMA forecasting was developed and evaluated "
            "specifically for Rayasandra Lake (STN 4531)."
        )

        forecast_copy = (
            forecast.copy()
        )

        forecast_copy[
            "Date"
        ] = pd.to_datetime(
            forecast_copy[
                "Date"
            ]
        )

        st.dataframe(
            forecast_copy[
                [
                    "Date",
                    "Forecast_HAB_Risk",
                    "Lower_CI",
                    "Upper_CI",
                    "Risk_Class",
                    "DSS_Action"
                ]
            ],
            use_container_width=True
        )

        forecast_chart = (
            forecast_copy[
                [
                    "Date",
                    "Forecast_HAB_Risk"
                ]
            ]
            .set_index(
                "Date"
            )
        )

        st.line_chart(
            forecast_chart
        )

        st.caption(
            "The ARIMA forecast is exploratory because the "
            "available monthly time series is relatively short."
        )

    else:

        st.info(
            "📈 ARIMA forecasting is currently available only "
            "for Rayasandra Lake (STN 4531), where the temporal "
            "model was specifically developed and evaluated."
        )


# ============================================================
# NEW SAMPLE PREDICTION
# ============================================================

elif page == "🧪 New Sample Prediction":

    st.header(
        "🧪 New Water-Quality Sample Prediction"
    )

    st.write(
        """
        Enter new or previously unseen water-quality values.

        The system will:

        1. Calculate the current proxy-derived HAB risk score.
        2. Assign a relative risk class.
        3. Predict next-month Lower or Elevated proxy risk.
        4. Explain the machine-learning prediction using SHAP.
        5. Generate a DSS recommendation.
        """
    )

    values = {}

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT SIDE INPUTS
    # --------------------------------------------------------

    with col1:

        values["DO"] = st.number_input(
            "Dissolved Oxygen (DO)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        values["pH"] = st.number_input(
            "pH",
            min_value=0.0,
            value=7.5,
            step=0.1
        )

        values["BOD"] = st.number_input(
            "BOD",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        values["COD"] = st.number_input(
            "COD",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

    # --------------------------------------------------------
    # RIGHT SIDE INPUTS
    # --------------------------------------------------------

    with col2:

        values["Nitrate"] = st.number_input(
            "Nitrate",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        values["Ammonical_N"] = st.number_input(
            "Ammonical-N",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        values["Phosphate"] = st.number_input(
            "Phosphate",
            min_value=0.0,
            value=0.5,
            step=0.1
        )

        values["Turbidity"] = st.number_input(
            "Turbidity",
            min_value=0.0,
            value=5.0,
            step=0.1
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "Predict HAB Risk",
        type="primary"
    ):

        # ----------------------------------------------------
        # RANGE CHECK
        # ----------------------------------------------------

        warnings = (
            check_ranges(
                values
            )
        )

        if warnings:

            st.warning(
                "Some entered values are outside the historical "
                "KSPCB range. Prediction reliability may be reduced."
            )

            for item in warnings:

                st.write(
                    "•",
                    item
                )


        # ----------------------------------------------------
        # CURRENT HAB PROXY SCORE
        # ----------------------------------------------------

        proxy_score = (
            calculate_proxy_score(
                values
            )
        )

        risk_class = (
            get_risk_class(
                proxy_score
            )
        )


        # ----------------------------------------------------
        # CREATE ML INPUT
        # ----------------------------------------------------

        new_sample = pd.DataFrame(
            [[
                values[f]
                for f in features
            ]],
            columns=features
        )


        # ----------------------------------------------------
        # XGBOOST PREDICTION
        # ----------------------------------------------------

        prediction = (
            model.predict(
                new_sample
            )[0]
        )

        probability = (
            model.predict_proba(
                new_sample
            )[0, 1]
        )

        prediction_label = (
            "Elevated Risk"
            if prediction == 1
            else "Lower Risk"
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.subheader(
            "Decision Support Result"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Current Proxy Score",
                f"{proxy_score:.2f}/100"
            )

        with r2:

            st.metric(
                "Current Risk Class",
                risk_class
            )

        with r3:

            st.metric(
                "Next-Month ML Warning",
                prediction_label
            )

        st.metric(
            "Elevated-Risk Probability",
            f"{probability * 100:.2f}%"
        )

        if prediction == 1:

            st.warning(
                "⚠️ The machine-learning model predicts "
                "Elevated proxy-risk conditions for the next month."
            )

        else:

            st.success(
                "The machine-learning model predicts "
                "Lower proxy-risk conditions for the next month."
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.subheader(
            "🔍 Why did the model make this prediction?"
        )

        shap_values = explainer.shap_values(
            new_sample
        )

        if isinstance(
            shap_values,
            list
        ):

            shap_values_current = (
                shap_values[-1][0]
            )

        else:

            shap_values_current = (
                np.array(
                    shap_values
                )[0]
            )

        driver_table = pd.DataFrame({

            "Feature":
                features,

            "Value":
                [
                    values[f]
                    for f in features
                ],

            "SHAP Value":
                shap_values_current
        })

        driver_table[
            "Absolute SHAP"
        ] = (
            driver_table[
                "SHAP Value"
            ].abs()
        )

        driver_table = (
            driver_table
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
        )

        st.dataframe(
            driver_table[
                [
                    "Feature",
                    "Value",
                    "SHAP Value"
                ]
            ]
            .head(5)
            .round(3),

            use_container_width=True
        )

        st.caption(
            "Positive SHAP values generally push the model "
            "toward Elevated Risk; negative values generally "
            "push the model toward Lower Risk."
        )

        st.caption(
            "SHAP values indicate influence on the machine-learning "
            "prediction and do not establish environmental causality."
        )


        # ====================================================
        # DSS RECOMMENDATION
        # ====================================================

        st.subheader(
            "📋 DSS Recommendation"
        )

        show_dss_box(
            risk_class
        )

        st.caption(
            "This dashboard supports environmental monitoring "
            "decisions and does not replace field sampling, "
            "laboratory verification, or regulatory assessment."
        )