import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

DB_PATH = "logs/violations.db"
REPORT_CSV = "logs/violations_report.csv"

def export_to_csv():
    """Export all violations from SQLite to CSV."""
    if not os.path.exists(DB_PATH):
        print("⚠️  No database found. Run detection first.")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    if df.empty:
        print("✅ No violations to export.")
        return

    df.to_csv(REPORT_CSV, index=False)
    print(f"✅ Violations exported to {REPORT_CSV}")

def visualize_data():
    """Show basic stats and graphs."""
    if not os.path.exists(REPORT_CSV):
        print("⚠️  Run export_to_csv() first.")
        return

    df = pd.read_csv(REPORT_CSV)

    # Violations per type
    type_counts = df["type"].value_counts()

    plt.figure(figsize=(6,4))
    type_counts.plot(kind='bar', color='crimson', edgecolor='black')
    plt.title("Violations by Type")
    plt.xlabel("Violation Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    export_to_csv()
    visualize_data()
