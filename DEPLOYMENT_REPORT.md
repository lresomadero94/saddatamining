# Deployment Report

## 1. Reused midterm work
The Streamlit application reuses the midterm reservation-outcome definition, cleaning logic, feature engineering, and Decision Tree configuration.

## 2. Changes for deployment
- Added Streamlit navigation.
- Added CSV schema validation and data-quality summaries.
- Added interactive Plotly EDA.
- Added saved-model inference.
- Added administrator dashboard and review guidance.
- Added SAD requirements, actors, use-case diagram, and analytical workflow.

## 3. Testing checklist
- CSV upload with valid schema
- CSV upload with missing columns
- Empty CSV
- Duplicate/null summary
- Three or more interactive charts
- Valid prediction inputs
- Invalid prediction inputs
- Saved model loading
- Separation of observed facts and predictions
- Human-review limitation displayed
- All modules accessible

## 4. Known limitation
The original dataset and exported model artifact were not included with the uploaded notebook, so the final `trained_model.joblib` must be produced using the original supplied synthetic dataset before deployment.
