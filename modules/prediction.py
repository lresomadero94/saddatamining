from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

PREDICTORS = [
    "Facility_Type", "User_Category", "Participants",
    "Duration_Hours", "Lead_Days", "Weekday", "Start_Hour",
    "Facility_Capacity"
]

def load_model(path):
    return joblib.load(path)

def model_info(model):
    return {
        "classes": list(getattr(model, "classes_", [])),
        "has_probability": hasattr(model, "predict_proba")
    }

def render_prediction(model):
    st.markdown("### Saved model")
    info = model_info(model)
    st.write("Classes:", info["classes"])
    st.write("Probability output supported:", info["has_probability"])

    with st.form("prediction_form"):
        facility = st.text_input("Facility Type", "Classroom")
        user = st.text_input("User Category", "Student")
        participants = st.number_input("Participants", min_value=1, value=20)
        duration = st.number_input("Duration Hours", min_value=0.25, value=1.0, step=0.25)
        lead = st.number_input("Lead Days", min_value=0, value=7)
        weekday = st.selectbox("Weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        start_hour = st.number_input("Start Hour", min_value=0, max_value=23, value=9)
        capacity = st.number_input("Facility Capacity", min_value=1, value=50)
        submitted = st.form_submit_button("Predict Reservation Outcome")

    if not submitted:
        return

    row = pd.DataFrame([{
        "Facility_Type": facility,
        "User_Category": user,
        "Participants": participants,
        "Duration_Hours": duration,
        "Lead_Days": lead,
        "Weekday": weekday,
        "Start_Hour": start_hour,
        "Facility_Capacity": capacity
    }])

    try:
        pred = model.predict(row)[0]
        st.success(f"Predicted outcome: {pred}")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(row)[0]
            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probs
            })
            st.dataframe(prob_df, use_container_width=True)
        st.caption(
            f"Prediction generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
            "This is model output, not an automatic authorization decision."
        )
    except Exception as e:
        st.error(f"Prediction failed. Check that inputs match the saved pipeline: {e}")
