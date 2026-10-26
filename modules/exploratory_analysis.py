import pandas as pd
import plotly.express as px
import streamlit as st

def render_eda(df):
    st.title("Exploratory Data Analysis")
    st.markdown("### Descriptive statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("### Class distribution")
    if "Reservation_Outcome" in df.columns:
        counts = df["Reservation_Outcome"].value_counts(dropna=False).reset_index()
        counts.columns = ["Outcome", "Count"]
        st.plotly_chart(
            px.bar(counts, x="Outcome", y="Count", title="Reservation Outcome Distribution"),
            use_container_width=True
        )

    st.markdown("### Interactive charts")
    c1, c2 = st.columns(2)
    with c1:
        if "Facility_Type" in df.columns:
            fc = df["Facility_Type"].value_counts().reset_index()
            fc.columns = ["Facility_Type", "Reservations"]
            st.plotly_chart(
                px.bar(fc, x="Facility_Type", y="Reservations",
                       title="Reservation Count by Facility Type"),
                use_container_width=True
            )
    with c2:
        if {"User_Category", "Reservation_Outcome"}.issubset(df.columns):
            t = df.dropna(subset=["User_Category", "Reservation_Outcome"]).copy()
            t["Risk_Event"] = t["Reservation_Outcome"].eq("Not Completed")
            us = t.groupby("User_Category").agg(
                Total=("Risk_Event", "size"),
                Not_Completed=("Risk_Event", "sum")
            ).reset_index()
            us["Not_Completed_Percent"] = 100 * us["Not_Completed"] / us["Total"]
            st.plotly_chart(
                px.bar(us, x="User_Category", y="Not_Completed_Percent",
                       title="Cancellation / No-Show Rate by User Category",
                       labels={"Not_Completed_Percent": "Percent"}),
                use_container_width=True
            )

    if {"Lead_Time_Category", "Reservation_Outcome"}.issubset(df.columns):
        lt = pd.crosstab(
            df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index"
        ).reset_index()
        lt = lt.melt(id_vars="Lead_Time_Category", var_name="Outcome", value_name="Percent")
        lt["Percent"] *= 100
        st.plotly_chart(
            px.bar(lt, x="Lead_Time_Category", y="Percent", color="Outcome",
                   barmode="stack", title="Reservation Outcome by Lead-Time Category",
                   labels={"Percent": "Percentage of Reservations"}),
            use_container_width=True
        )
