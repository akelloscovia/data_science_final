"""
Application Configuration

Stores settings used by Flask.
"""

class Config:

    # Secret key protects sessions
    SECRET_KEY = "goat-monitor-secret-key"

    # SQLite database
    SQLALCHEMY_DATABASE_URI = "sqlite:///goat_monitor.db"

    # Disable unnecessary tracking
    SQLALCHEMY_TRACK_MODIFICATIONS = False