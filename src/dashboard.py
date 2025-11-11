import streamlit as st
import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# --- CONFIG ---
st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")

DB_PATH = "logs/violations.db"

# --- AUTO REFRESH EVERY 30s ---
st_autorefresh(interval=30000, key="refresh_dashboard")

# --- HEADER ---
st.title("🚦 Traffic Violation Detection Dashboard")
st.markdown("#### Real-time traffic analytics using YOLOv8, OpenCV & SQLite")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Dashboard Controls")
refresh = st.sidebar.button("🔄 Refresh Data")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()
    return df

df = load_data()

if df.empty:
    st.warning("⚠️ No violations recorded yet. Run detection script first.")
else:
    # --- SIDEBAR FILTERS ---
    violation_types = ["All"] + sorted(df["type"].unique().tolist())
    selected_type = st.sidebar.selectbox("Filter by Violation Type", violation_types)

    if selected_type != "All":
        df = df[df["type"] == selected_type]

    # --- METRICS CARDS ---
    total = len(df)
    red_light = len(df[df["type"] == "Red Light"])
    overspeed = len(df[df["type"] == "Overspeed"])
    other = len(df[(df["type"] != "Red Light") & (df["type"] != "Overspeed")])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚗 Total Violations", total)
    col2.metric("🔴 Red Light", red_light)
    col3.metric("💨 Overspeed", overspeed)
    col4.metric("⚠️ Other", other)

    # --- CHARTS SECTION ---
    st.subheader("📊 Violation Analytics")
    col_chart1, col_chart2 = st.columns(2)

    # Violations by type
    with col_chart1:
        st.markdown("**Violations by Type**")
        type_counts = df["type"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        type_counts.plot(kind="bar", color="crimson", ax=ax)
        plt.xlabel("Violation Type")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    # Violations over time
    with col_chart2:
        st.markdown("**Violations Over Time**")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        daily_counts = df.groupby(df["timestamp"].dt.date)["id"].count()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        daily_counts.plot(kind="line", marker="o", color="darkblue", ax=ax2)
        plt.xlabel("Date")
        plt.ylabel("Violations")
        plt.tight_layout()
        st.pyplot(fig2)

    # --- DATA TABLE ---
    st.subheader("📋 Violation Records")
    st.dataframe(df, use_container_width=True)

    # --- IMAGE GALLERY ---
    st.subheader("🖼️ Violation Snapshots")
    img_cols = st.columns(3)
    for idx, (_, row) in enumerate(df.iterrows()):
        with img_cols[idx % 3]:
            if os.path.exists(row["image_path"]):
                st.image(
                    row["image_path"],
                    caption=f"{row['type']} | Vehicle {row['vehicle_id']}",
                    width=250,
                )

    # --- DOWNLOAD BUTTON ---
    st.subheader("📥 Export Report")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Violations Report (CSV)",
        data=csv,
        file_name="violations_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "Developed by **Siddhant** | JIIT Noida 🧠 | Powered by YOLOv8 + Streamlit + SQLite",
    unsafe_allow_html=True,
)
