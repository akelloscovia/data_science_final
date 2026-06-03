from run import app
from app import db
from app.models.user import User
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash

with app.app_context():
    # Ensure admin exists
    if User.query.filter_by(username="admin").first() is None:
        admin = User(username="admin", password=generate_password_hash("admin123"))
        db.session.add(admin)
        db.session.commit()
    # Print existing users for debugging
    print("Existing users:")
    for u in User.query.all():
        print(f"- {u.username}: {u.password}")

    # Directly test password verification
    admin = User.query.filter_by(username="admin").first()
    if admin:
        ok = check_password_hash(admin.password, "admin123")
        print("Direct check_password_hash(admin,password) ->", ok)

    # Generate a fresh hash and verify it immediately
    fresh = generate_password_hash("admin123")
    print("Fresh hash:", fresh)
    print("Verify fresh hash ->", check_password_hash(fresh, "admin123"))

    client = app.test_client()

    # Ensure admin password is reset for testing
    client.get("/bootstrap-admin?force=1")

    resp = client.post("/", data={"username": "admin", "password": "admin123"}, follow_redirects=True)

    print("Status code:", resp.status_code)
    print("Response data snippet:")
    print(resp.data.decode()[:1000])
