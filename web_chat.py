# web_chat.py
"""Flask blueprint for BrainNet chat interface."""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify, render_template, request

from brainnet.session_store import SessionManager

chat_bp = Blueprint("chat", __name__)
_session_manager = SessionManager(max_sessions=10, timeout_seconds=1800)


def _check_api_key() -> str | None:
    """Return an error message if ANTHROPIC_API_KEY is not set."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "ANTHROPIC_API_KEY is not set. "
            "Set this environment variable to enable the chat feature."
        )
    return None


@chat_bp.route("/chat")
def chat_page():
    api_error = _check_api_key()
    return render_template("chat.html", api_error=api_error)


@chat_bp.route("/chat/new", methods=["POST"])
def new_session():
    try:
        session = _session_manager.create_session()
        return jsonify({"session_id": session.session_id})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 429


@chat_bp.route("/chat/send", methods=["POST"])
def send_message():
    api_error = _check_api_key()
    if api_error:
        return jsonify({"error": api_error}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    session_id = data.get("session_id")
    message = data.get("message", "").strip()
    if not session_id or not message:
        return jsonify({"error": "session_id and message required"}), 400

    session = _session_manager.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found or expired"}), 404

    from brainnet.orchestrator import run_orchestrator

    def generate():
        try:
            for event in run_orchestrator(session, message):
                yield event
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@chat_bp.route("/chat/history")
def chat_history():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    session = _session_manager.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"messages": session.messages})


@chat_bp.route("/chat/status")
def chat_status():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    session = _session_manager.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "session_id": session.session_id,
        "n_strategies": len(session.strategies),
        "n_roi_datasets": len(session.roi_data),
        "estimated_tokens": session.estimate_tokens(),
    })
