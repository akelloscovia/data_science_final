from datetime import datetime
from app import db


class FeedItem(db.Model):
    __tablename__ = "feed_items"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    quantity_kg = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(50), default="Available")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<FeedItem {self.name} {self.quantity_kg}kg>"
