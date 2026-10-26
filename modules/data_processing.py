import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Facility_Type", "User_Category", "Usage_Status",
    "Participants", "Actual_Attendance", "Lead_Time_Days",
    "Cancellation_Days_Before", "Duration_Hours", "Weekday",
    "Start_Hour", "Facility_Capacity"
]

PREDICTOR_COLUMNS = [
    "Facility_Type", "User_Category", "Participants",
    "Duration_Hours", "Lead_Days", "Weekday", "Start_Hour",
    "Facility_Capacity"
]

def validate_schema(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return missing, extra

def clean_data(df):
    df = df.copy(deep=True)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for col in ["Facility_Type", "User_Category", "Usage_Status"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.title()

    if "Usage_Status" in df.columns:
        df["Usage_Status"] = df["Usage_Status"].replace({
            "No Show": "No-Show",
            "Noshow": "No-Show",
            "No-Show": "No-Show",
            "Used": "Used",
            "Cancelled": "Cancelled",
            "Canceled": "Cancelled"
        })

    for col in ["Participants", "Actual_Attendance", "Lead_Time_Days",
                "Cancellation_Days_Before", "Duration_Hours",
                "Start_Hour", "Facility_Capacity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Participants", "Actual_Attendance"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    if "Participants" in df.columns:
        df.loc[df["Participants"] == 0, "Participants"] = np.nan
    if "Lead_Time_Days" in df.columns:
        df.loc[df["Lead_Time_Days"] < 0, "Lead_Time_Days"] = np.nan
    if "Cancellation_Days_Before" in df.columns:
        df.loc[df["Cancellation_Days_Before"] < 0, "Cancellation_Days_Before"] = np.nan

    if {"Actual_Attendance", "Participants"}.issubset(df.columns):
        bad = (
            df["Actual_Attendance"].notna() &
            df["Participants"].notna() &
            (df["Actual_Attendance"] > df["Participants"])
        )
        df.loc[bad, "Actual_Attendance"] = np.nan

    if "Lead_Time_Days" in df.columns:
        conditions = [
            df["Lead_Time_Days"].between(0, 2),
            df["Lead_Time_Days"].between(3, 7),
            df["Lead_Time_Days"].between(8, 14),
            df["Lead_Time_Days"] > 14
        ]
        df["Lead_Time_Category"] = np.select(
            conditions,
            ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"],
            default="Unknown"
        )
        df.loc[df["Lead_Time_Days"].isna(), "Lead_Time_Category"] = pd.NA

    if "Usage_Status" in df.columns:
        status = df["Usage_Status"].astype("string").str.strip().str.lower()
        df["Reservation_Outcome"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df.loc[status == "used", "Reservation_Outcome"] = "Completed"
        df.loc[status.isin(["cancelled", "no-show", "no show", "noshow"]),
               "Reservation_Outcome"] = "Not Completed"

    if {"Actual_Attendance", "Participants"}.issubset(df.columns):
        denom_ok = df["Participants"].notna() & (df["Participants"] > 0)
        df["Attendance_Ratio"] = np.nan
        df.loc[denom_ok, "Attendance_Ratio"] = (
            df.loc[denom_ok, "Actual_Attendance"] / df.loc[denom_ok, "Participants"]
        )

    return df
