"""
User Model
----------

This file defines the User table.

The User table stores:
- username
- password (hashed)
- role

Flask-Login uses UserMixin to manage user sessions.
"""

from app import db
from flask_login import UserMixin


class User(UserMixin, db.Model):
    """
    User Database Model
    """

    __tablename__ = "users"

    # Primary Key
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # Username
    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    # Hashed Password
    password = db.Column(
        db.String(255),
        nullable=False
    )

    # User Role
    role = db.Column(
        db.String(50),
        default="admin"
    )

    def __repr__(self):
        """
        Used when printing user object
        """
        return f"<User {self.username}>"