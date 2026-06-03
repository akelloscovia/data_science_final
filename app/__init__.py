"""
Application Initialization
--------------------------

This file creates and configures the Flask application.

Responsibilities:
1. Create Flask app
2. Connect Database
3. Configure Flask-Login
4. Create database tables
"""

from pathlib import Path

import pandas as pd
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Database object
db = SQLAlchemy()

# Login manager object
login_manager = LoginManager()

DATASET_PATH = Path(__file__).resolve().parent.parent / "machine_learning" / "dataset" / "animal_activity_sample.csv"


def seed_database():
    from app.models.animal import Animal
    from app.models.activity import ActivityRecord
    from app.models.feed import FeedItem
    from app.models.user import User
    from werkzeug.security import generate_password_hash

    # Always ensure admin user exists (for Render deployment)
    if User.query.filter_by(username="admin").first() is None:
        admin = User(
            username="admin",
            password=generate_password_hash("admin123"),
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()

    if Animal.query.first() is not None:
        return

    if not DATASET_PATH.exists():
        return

    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=[col for col in ["Unnamed: 0"] if col in df.columns], errors="ignore")
    df = df.dropna(subset=["animal_ID"]).reset_index(drop=True)

    animal_map = {}
    for animal_code in df["animal_ID"].unique():
        animal = Animal(animal_id=str(animal_code))
        db.session.add(animal)
        animal_map[animal_code] = animal

    db.session.commit()

    records = []
    for _, row in df.iterrows():
        animal = animal_map.get(row["animal_ID"])
        if animal is None:
            continue

        records.append(ActivityRecord(
            animal_id=animal.id,
            segment_id=int(row["segment_ID"]) if not pd.isna(row["segment_ID"]) else None,
            timestamp_ms=int(row["timestamp_ms"]) if not pd.isna(row["timestamp_ms"]) else None,
            ax=float(row["ax"]) if not pd.isna(row["ax"]) else None,
            ay=float(row["ay"]) if not pd.isna(row["ay"]) else None,
            az=float(row["az"]) if not pd.isna(row["az"]) else None,
            gx=float(row["gx"]) if not pd.isna(row["gx"]) else None,
            gy=float(row["gy"]) if not pd.isna(row["gy"]) else None,
            gz=float(row["gz"]) if not pd.isna(row["gz"]) else None,
            label=str(row["label"]) if not pd.isna(row["label"]) else None,
        ))

    db.session.bulk_save_objects(records)

    default_feeds = [
        FeedItem(name="Grass Mix", quantity_kg=1800.0, status="Available"),
        FeedItem(name="Pellet Feed", quantity_kg=1250.0, status="Available"),
    ]
    db.session.add_all(default_feeds)
    db.session.commit()


def create_app():
    """
    Application Factory

    Creates and returns the Flask app.
    """

    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )

    # Load configuration settings
    app.config.from_object("config.Config")

    # Initialize database
    db.init_app(app)

    # Initialize login manager
    login_manager.init_app(app)

    # Redirect unauthenticated users
    login_manager.login_view = "login"

    # Import User model
    from app.models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        """
        Loads a user from database
        using their ID.
        """
        return User.query.get(int(user_id))

    # Create database tables and seed data
    with app.app_context():
        db.create_all()
        seed_database()

    return app