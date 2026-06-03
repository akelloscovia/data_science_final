"""
Main Application File
---------------------

Handles:
- Login
- Dashboard
- Logout

Application Entry Point
"""

from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from flask import flash

import math

from flask_login import login_user
from flask_login import logout_user
from flask_login import login_required
from flask_login import current_user

from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from app import create_app, db
from app.models import Animal, ActivityRecord, FeedItem
from app.models.user import User

# Create Flask app
app = create_app()


@app.route("/", methods=["GET", "POST"])
def login():
    """
    Login Page

    GET:
        Display login form

    POST:
        Validate credentials
    """

    if request.method == "POST":

        # Get values from form
        username = request.form.get("username")
        password = request.form.get("password")

        # Search for user
        user = User.query.filter_by(
            username=username
        ).first()

        # Verify password
        if user and check_password_hash(
            user.password,
            password
        ):

            # Log user in
            login_user(user)

            return redirect(
                url_for("dashboard")
            )

        return """
        <h3 style='color:red'>
        Invalid Username or Password
        </h3>
        """

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """
    Dashboard Page

    Requires Login
    """

    total_animals = Animal.query.count()
    total_activity = ActivityRecord.query.count()
    total_feed = sum(item.quantity_kg or 0 for item in FeedItem.query.all())
    recent_activity = ActivityRecord.query.order_by(ActivityRecord.timestamp_ms.desc()).limit(6).all()

    return render_template(
        "dashboard.html",
        total_animals=total_animals,
        total_activity=total_activity,
        total_feed=total_feed,
        recent_activity=recent_activity,
    )


@app.route("/goats", methods=["GET", "POST"])
@login_required
def goats():
    """
    Goats management page.
    """

    message = None
    error = None

    if request.method == "POST":
        animal_id = request.form.get("animal_id", "").strip()
        if not animal_id:
            error = "Animal ID is required."
        elif Animal.query.filter_by(animal_id=animal_id).first():
            error = f"Animal '{animal_id}' already exists."
        else:
            new_animal = Animal(animal_id=animal_id)
            db.session.add(new_animal)
            db.session.commit()
            message = f"Animal '{animal_id}' added successfully."

    animals = Animal.query.order_by(Animal.animal_id).all()
    return render_template("goats.html", animals=animals, message=message, error=error)


@app.route("/health")
@login_required
def health():
    """
    Health records page.
    """

    label_summary = (
        db.session.query(
            ActivityRecord.label,
            db.func.count(ActivityRecord.id),
        )
        .group_by(ActivityRecord.label)
        .all()
    )
    recent_records = ActivityRecord.query.order_by(ActivityRecord.timestamp_ms.desc()).limit(12).all()

    return render_template(
        "health.html",
        label_summary=label_summary,
        recent_records=recent_records,
    )


@app.route("/health/edit/<int:record_id>", methods=["GET", "POST"])
@login_required
def edit_health(record_id):
    """
    Edit a single health/activity record.
    """

    record = ActivityRecord.query.get_or_404(record_id)

    if request.method == "POST":
        label = request.form.get("label")
        try:
            record.ax = float(request.form.get("ax")) if request.form.get("ax") not in (None, "") else None
            record.ay = float(request.form.get("ay")) if request.form.get("ay") not in (None, "") else None
            record.az = float(request.form.get("az")) if request.form.get("az") not in (None, "") else None
            record.gx = float(request.form.get("gx")) if request.form.get("gx") not in (None, "") else None
        except Exception:
            # ignore conversion errors and preserve existing values
            pass

        try:
            record.gy = float(request.form.get("gy")) if request.form.get("gy") not in (None, "") else None
        except Exception:
            pass

        try:
            record.gz = float(request.form.get("gz")) if request.form.get("gz") not in (None, "") else None
        except Exception:
            pass

        try:
            ts = request.form.get("timestamp_ms")
            record.timestamp_ms = int(ts) if ts not in (None, "") else record.timestamp_ms
        except Exception:
            pass

        if label is not None:
            record.label = label

        db.session.commit()

        return redirect(url_for("health"))

    return render_template("edit_record.html", record=record)


@app.route("/feed", methods=["GET", "POST"])
@login_required
def feed():
    """
    Feed management page.
    """

    if request.method == "POST":
        feed_id = request.form.get("feed_id")
        quantity = request.form.get("quantity_kg")
        item = FeedItem.query.get(feed_id)
        if item and quantity is not None:
            try:
                item.quantity_kg = float(quantity)
                item.status = "Available" if item.quantity_kg > 0 else "Out of stock"
                db.session.commit()
            except ValueError:
                pass

    feed_items = FeedItem.query.order_by(FeedItem.name).all()
    return render_template("feed.html", feed_items=feed_items)


@app.route("/predictions", methods=["GET", "POST"])
@login_required
def predictions():
    """
    Predictions page wired to the ML model.
    """

    from machine_learning.predict import predict

    prediction = None
    error = None
    sample_values = {
        "ax": 0.2,
        "ay": 0.1,
        "az": 0.05,
        "gx": 0.01,
        "gy": 0.02,
        "gz": 0.03,
    }

    if request.method == "POST":
        try:
            ax = float(request.form.get("ax", sample_values["ax"]))
            ay = float(request.form.get("ay", sample_values["ay"]))
            az = float(request.form.get("az", sample_values["az"]))
            gx = float(request.form.get("gx", sample_values["gx"]))
            gy = float(request.form.get("gy", sample_values["gy"]))
            gz = float(request.form.get("gz", sample_values["gz"]))

            acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
            gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
            features = [ax, ay, az, gx, gy, gz, acc_mag, gyro_mag]

            prediction = predict(features)[0]
        except ValueError:
            error = "Sensor values must be valid numbers."
        except FileNotFoundError:
            error = "Prediction artifacts are not available. Train the model first."
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "predictions.html",
        prediction=prediction,
        error=error,
        sample=sample_values
    )


@app.route("/logout")
@login_required
def logout():
    """
    Logout User
    """

    logout_user()

    return redirect(
        url_for("login")
    )


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    """
    Change the current user's password.
    """

    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not current_password or not new_password or not confirm_password:
            flash("All fields are required.", "danger")
        elif not check_password_hash(current_user.password, current_password):
            flash("Current password is incorrect.", "danger")
        elif new_password != confirm_password:
            flash("New passwords do not match.", "danger")
        elif len(new_password) < 6:
            flash("New password must be at least 6 characters long.", "danger")
        else:
            current_user.password = generate_password_hash(new_password)
            db.session.commit()
            flash("Password changed successfully.", "success")
            return redirect(url_for("dashboard"))

    return render_template("change_password.html")


if __name__ == "__main__":

    app.run(
        debug=True
    )