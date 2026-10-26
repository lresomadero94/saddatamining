# Campus Facility Reservation Monitoring — Streamlit

This project deploys the midterm Campus Facility Reservation Monitoring data-mining workflow using Streamlit.

## Midterm model reproduced

- Target: `Reservation_Outcome`
- `Completed`: `Usage_Status = Used`
- `Not Completed`: `Usage_Status = Cancelled` or `No-Show`
- Classifier: Decision Tree
- `max_depth=5`
- `min_samples_leaf=3`
- `class_weight="balanced"`
- `random_state=42`
- Train/test split: 80% / 20%
- Booking-time predictors:
  - `Facility_Type`
  - `User_Category`
  - `Participants`
  - `Duration_Hours`
  - `Lead_Days`
  - `Weekday`
  - `Start_Hour`
  - `Facility_Capacity`

Post-outcome fields such as `Usage_Status`, `Actual_Attendance`, and `Cancellation_Days_Before` are not used as prediction features.

## Run locally

```bash
pip install -r requirements.txt
python build_model.py --data facility_reservations.csv
streamlit run app.py
```

The `facility_reservations.csv` file is intentionally not fabricated or bundled here. Use the supplied synthetic dataset from the examination/midterm.

## Required repository structure

```text
app.py
requirements.txt
README.md
build_model.py
modules/
  data_processing.py
  exploratory_analysis.py
  prediction.py
  dashboard.py
models/
  trained_model.joblib
assets/
  use_case_diagram.png
```

## Streamlit modules

1. Home
2. Dataset
3. EDA
4. Prediction
5. Dashboard
6. System Design

## Important limitation

The uploaded notebook contains the training code but the actual `facility_reservations.csv` and exported `trained_model.joblib` were not included in the files available to this build. Therefore, the exact model artifact must be generated from the original supplied dataset before deployment. Do not substitute invented data.

Predictions are decision-support outputs and should not automatically authorize rejection of a reservation.
