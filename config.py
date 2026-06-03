"""
Application Configuration

Stores settings used by Flask.
"""

import os

class Config:

    # Secret key protects sessions
    SECRET_KEY = os.environ.get("SECRET_KEY", "goat-monitor-secret-key")

    # Use a database URL when deployed or fallback to SQLite locally
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///goat_monitor.db"
    )

    # Disable unnecessary tracking
    SQLALCHEMY_TRACK_MODIFICATIONS = False