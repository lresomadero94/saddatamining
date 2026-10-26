import pandas as pd
import plotly.express as px
import streamlit as st

def render_dashboard(df):
    st.title("Decision-Support Dashboard")
    st.caption("Observed facts and model predictions must be interpreted separately.")

    total = len(df)
    completed = int((df["Reservation_Outcome"] == "Completed").sum()) if "Reservation_Outcome" in df else 0
    not_completed = int((df["Reservation_Outcome"] == "Not Completed").sum()) if "Reservation_Outcome" in df else 0
    rate = (not_completed / total * 100) if total else 0

    a,b,c,d = st.columns(4)
    a.metric("Total Reservations", total)
    b.metric("Completed", completed)
    c.metric("Not Completed", not_completed)
    d.metric("Not Completed Rate", f"{rate:.1f}%")

    c1,c2 = st.columns(2)
    with c1:
        if "Facility_Type" in df.columns:
            x = df["Facility_Type"].value_counts().reset_index()
            x.columns = ["Facility_Type","Reservations"]
            st.plotly_chart(px.bar(x, x="Facility_Type", y="Reservations",
                                   title="Reservations by Facility Type"),
                            use_container_width=True)
    with c2:
        if {"Lead_Time_Category","Reservation_Outcome"}.issubset(df.columns):
            x = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index").reset_index()
            x = x.melt(id_vars="Lead_Time_Category", var_name="Outcome", value_name="Percent")
            x["Percent"] *= 100
            st.plotly_chart(px.bar(x, x="Lead_Time_Category", y="Percent", color="Outcome",
                                   barmode="stack", title="Outcome by Lead-Time Category"),
                            use_container_width=True)

    st.markdown("### Administrator review queue")
    if {"Reservation_Outcome","Participants","Lead_Time_Category","Facility_Type","User_Category"}.issubset(df.columns):
        queue = df[df["Reservation_Outcome"] == "Not Completed"].copy()
        queue = queue.sort_values(["Lead_Time_Category","Participants"], ascending=[True,False])
        st.write("Historical Not Completed cases are shown here as observed review candidates; they are not model predictions.")
        cols = [c for c in ["Reservation_ID","Facility_Type","User_Category","Participants","Lead_Time_Days","Lead_Time_Category","Reservation_Outcome"] if c in queue.columns]
        st.dataframe(queue[cols].head(20), use_container_width=True)

    st.warning(
        "Recommended follow-up: review high-risk or problematic cases, verify the reservation context, "
        "and use reminders/confirmation or other appropriate administrative action. Do not automatically reject "
        "a reservation based only on a prediction."
    )
