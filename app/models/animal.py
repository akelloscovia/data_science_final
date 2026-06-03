from datetime import datetime
from app import db


class Animal(db.Model):
    __tablename__ = "animals"

    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.String(64), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    activity_records = db.relationship(
        "ActivityRecord",
        backref="animal",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )

    def latest_activity(self):
        from app.models.activity import ActivityRecord

        return self.activity_records.order_by(
            ActivityRecord.timestamp_ms.desc()
        ).first()

    @property
    def last_activity_label(self):
        latest = self.latest_activity()
        return latest.label if latest else "unknown"

    @property
    def record_count(self):
        return self.activity_records.count()

    def __repr__(self):
        return f"<Animal {self.animal_id}>"
