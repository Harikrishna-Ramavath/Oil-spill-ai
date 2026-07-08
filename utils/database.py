"""
utils/database.py

This module manages the SQLite database integration. It handles user creation,
authentication, saving prediction entries, and fetching metrics and logs for
the analytics dashboard.
"""

import os
import sqlite3
import json
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

# Helper function to get database connection
def get_db_connection():
    config = load_config()
    db_path = config["db_path"]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize Database Schema
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT NOT NULL,
        role TEXT DEFAULT 'operator',
        created_at TEXT NOT NULL
    )
    """)
    
    # Create Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_name TEXT NOT NULL,
        prediction_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        severity TEXT NOT NULL,
        spill_percentage REAL NOT NULL,
        original_image_path TEXT NOT NULL,
        heatmap_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

# Password Hashing Utilities
def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with a random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    # Store salt and key combined as hex
    return (salt + key).hex()

def verify_password(password: str, hashed_db_val: str) -> bool:
    """Verify password against pbkdf2 hash."""
    try:
        db_val_bytes = bytes.fromhex(hashed_db_val)
        salt = db_val_bytes[:16]
        stored_key = db_val_bytes[16:]
        new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return new_key == stored_key
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

# User Operations
def create_user(username, password, email, role='operator'):
    conn = get_db_connection()
    cursor = conn.cursor()
    pw_hash = hash_password(password)
    created_at = datetime.now().isoformat()
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, pw_hash, email, role, created_at)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and verify_password(password, user["password_hash"]):
        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
    return None

# Prediction Operations
def save_prediction(user_id, image_name, label, confidence, severity, spill_percentage, original_path, heatmap_path):
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()
    
    cursor.execute("""
    INSERT INTO predictions 
    (user_id, image_name, prediction_label, confidence, severity, spill_percentage, original_image_path, heatmap_path, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, image_name, label, confidence, severity, spill_percentage, original_path, heatmap_path, created_at))
    
    conn.commit()
    conn.close()

def get_prediction_history(user_id=None, date_from=None, date_to=None):
    """
    Fetches prediction records with filters.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT p.*, u.username FROM predictions p LEFT JOIN users u ON p.user_id = u.id WHERE 1=1"
    params = []
    
    if user_id:
        query += " AND p.user_id = ?"
        params.append(user_id)
        
    if date_from:
        query += " AND date(p.created_at) >= date(?)"
        params.append(date_from)
        
    if date_to:
        query += " AND date(p.created_at) <= date(?)"
        params.append(date_to)
        
    query += " ORDER BY p.created_at DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert list of rows to dictionary list
    return [dict(row) for row in rows]

# Analytics & Dashboard Metrics
def get_dashboard_metrics():
    """
    Aggregates metrics for display on the dashboard cards.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total Predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    
    # Total Oil Spills
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'Oil Spill'")
    total_spills = cursor.fetchone()[0]
    
    # Total No Spills
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'No Spill'")
    total_no_spills = cursor.fetchone()[0]
    
    # Average Spill Percentage
    cursor.execute("SELECT AVG(spill_percentage) FROM predictions WHERE prediction_label = 'Oil Spill'")
    avg_spill_pct = cursor.fetchone()[0] or 0.0
    
    conn.close()
    
    return {
        "total_predictions": total_predictions,
        "total_spills": total_spills,
        "total_no_spills": total_no_spills,
        "avg_spill_pct": round(avg_spill_pct, 2)
    }

def get_daily_detections():
    """
    Groups predictions by day for the daily trend chart.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT date(created_at) as prediction_date, 
           COUNT(*) as total,
           SUM(CASE WHEN prediction_label = 'Oil Spill' THEN 1 ELSE 0 END) as spills
    FROM predictions 
    GROUP BY prediction_date 
    ORDER BY prediction_date ASC
    LIMIT 30
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_severity_distribution():
    """
    Counts instances of each severity category.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT severity, COUNT(*) as count 
    FROM predictions 
    WHERE prediction_label = 'Oil Spill'
    GROUP BY severity
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Autopopulate Admin user on first run
def seed_admin_user():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        # Default user is operator/admin
        create_user("admin", "admin123", "admin@oilspillai.gov", "admin")
        logger.info("Default admin user seeded: admin / admin123")
    conn.close()

# Auto init DB on import
init_db()
seed_admin_user()
