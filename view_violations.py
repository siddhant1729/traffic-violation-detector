import sqlite3
from tabulate import tabulate
import os

DB_PATH = "logs/violations.db"

def view_violations():
    """Display all violations in a clean table format."""
    if not os.path.exists(DB_PATH):
        print("⚠️  No database found. Run the detection script first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM violations")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        print(tabulate(
            rows,
            headers=["ID", "Vehicle ID", "Type", "Timestamp", "Image Path"],
            tablefmt="grid"
        ))
    else:
        print("✅ No violations recorded yet.")

if __name__ == "__main__":
    view_violations()
