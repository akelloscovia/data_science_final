"""
Application Configuration

Stores settings used by Flask.
"""

import os

class Config:

    # Secret key protects sessions
    SECRET_KEY = os.environ.get("SECRET_KEY", "goat-monitor-secret-key")

    # Use a database URL when deployed or fallback to SQLite locally
    database_uri = os.environ.get(
        "DATABASE_URL",
        "sqlite:///goat_monitor.db"
    )

    # SQLAlchemy 2.x prefers postgresql:// instead of deprecated postgres://
    if database_uri and database_uri.startswith("postgres://"):
        database_uri = database_uri.replace("postgres://", "postgresql://", 1)

    SQLALCHEMY_DATABASE_URI = database_uri

    # Disable unnecessary tracking
    SQLALCHEMY_TRACK_MODIFICATIONS = False