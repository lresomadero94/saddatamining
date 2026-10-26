import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from modules.data_processing import REQUIRED_COLUMNS, clean_data, validate_schema
from modules.exploratory_analysis import render_eda
from modules.prediction import load_model, model_info, render_prediction
from modules.dashboard import render_dashboard

st.set_page_config(
    page_title="Campus Facility Reservation Monitoring",
    page_icon="🏫",
    layout="wide"
)

MODEL_PATH = Path("models/trained_model.joblib")

@st.cache_data
def cached_clean_data(df):
    return clean_data(df)

@st.cache_resource
def cached_model():
    return load_model(MODEL_PATH)

def get_dataset():
    if "dataset" not in st.session_state:
        return None
    return st.session_state["dataset"]

def home():
    st.title("Campus Facility Reservation Monitoring")
    st.subheader("Streamlit Decision-Support System")
    st.write(
        "A deployment of the midterm reservation data-mining workflow. "
        "The system supports facility administrators by combining observed "
        "reservation data, exploratory analysis, and model-based risk alerts."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Target", "Reservation_Outcome")
    c2.metric("Model", "Decision Tree")
    c3.metric("Split", "80% / 20%")
    st.info(
        "Prediction labels are decision-support outputs. A high-risk prediction "
        "must not automatically reject a reservation."
    )
    st.markdown("### Workflow")
    st.markdown(
        "Historical reservations → data validation → cleaning/feature engineering "
        "→ EDA → Decision Tree prediction → administrator review → human decision"
    )
    st.markdown("### Midterm model design")
    st.write(
        "The midterm notebook defines Reservation_Outcome as Completed when "
        "Usage_Status is Used and Not Completed when the status is Cancelled or "
        "No-Show. Post-outcome fields are excluded from prediction."
    )

def dataset_page():
    st.title("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload facility reservation CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload the supplied synthetic reservation dataset to begin.")
        return
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        return
    missing, extra = validate_schema(raw)
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    if raw.empty:
        st.error("The uploaded dataset is empty.")
        return
    cleaned = cached_clean_data(raw)
    st.session_state["dataset"] = cleaned
    st.success("Dataset loaded and cleaned using the midterm rules.")
    a, b, c = st.columns(3)
    a.metric("Rows", len(cleaned))
    b.metric("Columns", len(cleaned.columns))
    c.metric("Duplicates remaining", int(cleaned.duplicated().sum()))
    st.markdown("### Preview")
    st.dataframe(cleaned.head(20), use_container_width=True)
    st.markdown("### Missing-value summary")
    st.dataframe(
        cleaned.isna().sum().rename("Missing Values").to_frame(),
        use_container_width=True
    )

def prediction_page():
    st.title("Predictive Modeling")
    if not MODEL_PATH.exists():
        st.warning(
            "The trained_model.joblib file is not included because the uploaded "
            "midterm package did not contain the reservation dataset needed to "
            "fit/export the exact pipeline. Run build_model.py with the supplied "
            "facility_reservations.csv first."
        )
        st.code("python build_model.py --data facility_reservations.csv")
        return
    try:
        model = cached_model()
        render_prediction(model)
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")

def system_design():
    st.title("System Design / SAD Integration")
    st.markdown("### Actors")
    st.write(
        "**Facility User** (Student / Faculty / Staff), **Facility Administrator**, "
        "and **System Manager** (IT / Operations)."
    )
    st.markdown("### Functional Requirements")
    fr = [
        ("FR-DM-01", "The system shall validate uploaded reservation data and show schema, null, duplicate, and preview information."),
        ("FR-DM-02", "The system shall analyze reservation outcomes by facility type, user category, and lead-time category."),
        ("FR-DM-03", "The system shall predict Reservation_Outcome using the saved Decision Tree pipeline and booking-time predictors."),
        ("FR-DM-04", "The system shall present predicted high-risk reservations for facility administrator review."),
        ("FR-DM-05", "The system shall distinguish observed facts from model predictions and record that predictions do not automatically authorize rejection."),
    ]
    st.table(pd.DataFrame(fr, columns=["ID", "Requirement"]))
    st.markdown("### Non-functional Requirements")
    nfr = [
        ("NFR-01", "Performance", "Use cached loading/transformations so the saved model is not retrained on page refresh."),
        ("NFR-02", "Reliability", "Reject malformed or schema-incompatible CSV uploads with clear error messages."),
        ("NFR-03", "Usability", "Present labels, metrics, risk explanations, limitations, and human-review guidance clearly."),
    ]
    st.table(pd.DataFrame(nfr, columns=["ID", "Category", "Requirement"]))
    st.markdown("### Use-case diagram")
    img = Path("assets/use_case_diagram.png")
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.info("Place the updated UML use-case diagram at assets/use_case_diagram.png.")
    st.markdown("### Analytical decision workflow")
    st.markdown(
        "1. Administrator uploads/loads reservation data.\n"
        "2. System validates schema and data quality.\n"
        "3. System cleans and derives the midterm variables.\n"
        "4. Administrator reviews observed statistics and EDA.\n"
        "5. Saved Decision Tree produces a reservation-outcome prediction.\n"
        "6. System highlights higher-risk cases for review.\n"
        "7. Administrator checks the reservation context and model explanation.\n"
        "8. Administrator makes the final authorized decision or follow-up action."
    )

def main():
    menu = st.sidebar.selectbox(
        "Module",
        ["Home", "Dataset", "EDA", "Prediction", "Dashboard", "System Design"]
    )
    st.sidebar.caption("Midterm model: Decision Tree / Reservation_Outcome")
    if menu == "Home":
        home()
    elif menu == "Dataset":
        dataset_page()
    elif menu == "EDA":
        df = get_dataset()
        if df is None:
            st.info("Upload a dataset in the Dataset module first.")
        else:
            render_eda(df)
    elif menu == "Prediction":
        prediction_page()
    elif menu == "Dashboard":
        df = get_dataset()
        if df is None:
            st.info("Upload a dataset in the Dataset module first.")
        else:
            render_dashboard(df)
    else:
        system_design()

if __name__ == "__main__":
    main()
