"""CLI utility to reset or create the admin user password.

Usage:
    python scripts/reset_admin.py --password <newpassword>

This is intended for deployments where the `/bootstrap-admin` route is disabled.
"""
import argparse
from run import app
from app import db
from app.models.user import User
from werkzeug.security import generate_password_hash


def reset_admin(password: str):
    with app.app_context():
        user = User.query.filter_by(username="admin").first()
        if not user:
            user = User(username="admin")

        user.password = generate_password_hash(password)
        user.role = user.role or "admin"

        db.session.add(user)
        db.session.commit()

        print("Admin user created/updated. Username: admin | Password: (hidden)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", "-p", required=True, help="New admin password")
    args = parser.parse_args()

    reset_admin(args.password)


if __name__ == "__main__":
    main()
