from datetime import datetime
from app import db


class ActivityRecord(db.Model):
    __tablename__ = "activity_records"

    id = db.Column(db.Integer, primary_key=True)
    animal_id = db.Column(db.Integer, db.ForeignKey("animals.id"), nullable=False)
    segment_id = db.Column(db.Integer, nullable=True)
    timestamp_ms = db.Column(db.Integer, nullable=True)
    ax = db.Column(db.Float, nullable=True)
    ay = db.Column(db.Float, nullable=True)
    az = db.Column(db.Float, nullable=True)
    gx = db.Column(db.Float, nullable=True)
    gy = db.Column(db.Float, nullable=True)
    gz = db.Column(db.Float, nullable=True)
    label = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ActivityRecord {self.label} {self.animal_id} {self.timestamp_ms}>"
