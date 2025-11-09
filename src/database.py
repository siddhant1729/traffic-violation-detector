# src/database.py
import sqlite3
import os
from datetime import datetime

DB_PATH = "logs/violations.db"

def init_db():
    """Create the database and table if not exists."""
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            type TEXT,
            timestamp TEXT,
            image_path TEXT
        )
    ''')

    conn.commit()
    conn.close()

def log_violation(vehicle_id, violation_type, image_path):
    """Insert new violation record into DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO violations (vehicle_id, type, timestamp, image_path)
        VALUES (?, ?, ?, ?)
    ''', (vehicle_id, violation_type, timestamp, image_path))

    conn.commit()
    conn.close()
    print(f"✅ Logged violation: Vehicle {vehicle_id} | {violation_type} | {timestamp}")
