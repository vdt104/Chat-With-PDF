from flask import Blueprint, g, request, Response, jsonify, stream_with_context, current_app
from app.web.hooks import login_required, load_model
from app.web.db.models import Pdf, Conversation
from app.chat import build_chat, ChatArgs
from app.chat.streamable.streamable import StreamableChat
from langchain_core.messages import AIMessage, HumanMessage

import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint("conversation", __name__, url_prefix="/api/conversations")

@bp.route("/", methods=["GET"])
@login_required
@load_model(Pdf, lambda r: r.args.get("pdf_id"))
def list_conversations(pdf):
    return [c.as_dict() for c in pdf.conversations]

@bp.route("/", methods=["POST"])
@login_required
@load_model(Pdf, lambda r: r.args.get("pdf_id"))
def create_conversation(pdf):
    conversation = Conversation.create(user_id=g.user.id, pdf_id=pdf.id)
    return conversation.as_dict()

@bp.route("/<string:conversation_id>/messages", methods=["POST"])
@login_required
@load_model(Conversation)
def create_message(conversation):
    input_text = request.json.get("input")
    streaming = request.args.get("stream", False)

    pdf = conversation.pdf

    chat_args = ChatArgs(
        conversation_id=conversation.id,
        pdf_id=pdf.id,
        streaming=streaming,
        metadata={
            "conversation_id": conversation.id,
            "user_id": g.user.id,
            "pdf_id": pdf.id,
        },
    )

    chat = build_chat(chat_args)

    chat_history = conversation.get_chat_history()

    input_data = {"input": input_text, "chat_history": chat_history}

    config = {"configurable": {"thread_id": conversation.id}}

    if not chat:
        return "Chat not yet implemented!"

    if streaming:
        streamable_chat = StreamableChat(chat, input_data, config, conversation)
        return Response(stream_with_context(streamable_chat.stream()), mimetype="text/event-stream")
    else:
        response = chat.invoke(input_data, config=config)
        logger.info("Response from chat.invoke: %s", response)
        conversation.add_message(HumanMessage(content=input_text))
        conversation.add_message(AIMessage(content=response['answer']))
        return jsonify({"role": "assistant", "content": response['answer']})