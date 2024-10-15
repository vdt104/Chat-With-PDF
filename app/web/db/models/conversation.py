import uuid
from app.web.db import db
from .base import BaseModel
from app.web.db.models.message import Message  


class Conversation(BaseModel):
    id: str = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_on = db.Column(db.DateTime, server_default=db.func.now())

    retriever: str = db.Column(db.String)
    memory: str = db.Column(db.String)
    llm: str = db.Column(db.String)

    pdf_id: int = db.Column(db.Integer, db.ForeignKey("pdf.id"), nullable=False)
    pdf = db.relationship("Pdf", back_populates="conversations")

    user_id: int = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    user = db.relationship("User", back_populates="conversations")

    messages = db.relationship(
        "Message", back_populates="conversation", order_by="Message.created_on"
    )

    def as_dict(self):
        return {
            "id": self.id,
            "pdf_id": self.pdf_id,
            "messages": [m.as_dict() for m in self.messages],
        }

    def get_chat_history(self):
        messages = Message.query.filter_by(conversation_id=self.id).order_by(Message.created_on).all()
        return [message.as_lc_message() for message in messages]

    def add_message(self, message):
        new_message = Message(
            conversation_id=self.id,
            role=message.type,
            content=message.content
        )
        db.session.add(new_message)
        db.session.commit()
        return new_message