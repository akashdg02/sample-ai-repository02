"""
Chat Controller
Handles chat-related API endpoints.
"""

from flask import Blueprint, request, jsonify
from app.models.database import db,Conversation,Message
from app.models.ai_model import AIModel
from app.utils.helpers import generate_conversation_id, validate_request
import os

# Create Blueprint
chat_bp = Blueprint('chat', __name__)

# Initialise AI model
ai_model = AIModel()

# Health Endpoint
@chat_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running

    Returns:
        JSON: API Status
    """
    return jsonify({
        'status': 'healthy',
        'message':"Chatbot API is Running",
        'model': ai_model.get_mode_info()
    }), 200

@chat_bp.route('/chat', methods=['POST'])

def send_message():
    """
    Send Message to chatbot

    Request Body:
        {
            "message":"User Message",
            "conversation_id":"optional-conversation_id",
        }
    :return:
        JSON: AI Response and conversation metadata
    """
    try:
        data = request.get_json()

        if not data :
            return jsonify({
                'success':False,
                'error':'No JSON data provided.'
            }), 400

        if 'message' not in data or not data['message'].strip():
            return jsonify({
                'success': False,
                'error': 'Message field is Required.'
            }), 400

        user_message = data['message'].strip()
        conversation_id = data.get('conversation_id')

        if conversation_id:
            conversation = Conversation.query.get(conversation_id)
            if not conversation:
                conversation = Conversation(id=conversation_id)
                db.session.add(conversation)
        else:
            #Create new Conversation
            conversation_id = generate_conversation_id()
            conversation = Conversation(id=conversation_id)
            db.session.add(conversation)

        #User conversation save
        user_msg = Message(
            conversation_id=conversation.id,
            role='user',
            content=user_message
        )
        db.session.add(user_msg)
        db.session.flush()

        #BUild conversation history for AI
        messages = build_conversation_history(conversation.id)

        try:
            ai_response = ai_model.chat_with_history(messages)
        except Exception as e:
            db.session.rollback()
            return jsonify({
                'success':False,
                'error':f'AI Error: {str(e)}'
            }),500

        ai_msg = Message(
            conversation_id=conversation.id,
            role='assistant',
            content=ai_response,
            model=ai_model.model
        )

        db.session.add(ai_msg)

        if not conversation.title and user_message:
            conversation.title= user_message[:50] + ('...' if len(user_message) > 50 else '')

        db.session.commit()

        return jsonify({
            'status': True,
            'data':{
                'response': ai_response,
                'conversation_id': conversation.id,
                'message_id': ai_msg.id,
                'model':ai_model.model,
                'timestamp':ai_msg.timestamp.isoformat()
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@chat_bp.route('/conversations', methods=['GET'])
def get_conversations():
    """
    Get all conversations

    Query Parameters:
        limit(int): Max number of conversations (default 20)
        offset(int): Number of conversations to skip (default 0)
    :return:
        JSON: List of conversations
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)

        if limit < 1 or limit > 100:
            limit = 20;

        if offset < 0:
            offset = 0

        conversations = Conversation.query.order_by(Conversation.updated_at.desc()).limit(limit).offset(offset).all()

        total = Conversation.query.count()

        return jsonify({
            'success': True,
            'data': {
                'conversations': [conv.to_dict() for conv in conversations],
                'total': total,
                'limit': limit,
                'offset': offset
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 200

@chat_bp.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """
    Get Spepcific conversation with all messages

    Args:
        conversation_id (str) : Conversation ID

    Returns:
        JSON: Conversation data with messages
    """
    try:

        conversation = Conversation.query.get(conversation_id)

        if not conversation:
            return jsonify({
                'success':False,
                'error':'Conversation Not found'
            })

        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp.asc()).all()

        return jsonify({
            'success': True,
            'data': {
                'conversation_id':conversation.id,
                'title':conversation.title,
                'messages':[msg.to_dict() for msg in messages],
                'total_messages':len(messages),
                'created_at':conversation.created_at,
                'updated_at':conversation.updated_at
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 200

@chat_bp.route('/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """
    Delete Specific Conversation

    Args:
        conversation_id Conversation ID

    Returns
        JSON: success message
    """
    try:
        conversation = Conversation.query.get(conversation_id)

        if not conversation:
            return jsonify({
                'success':False,
                'error':'Conversation not found'
            }),404

            db.session.delete(conversation)
            db.session.commit()
        return jsonify({
            'success': True,
            'message': "Conversation deleted Successfuly"
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 200

@chat_bp.route('/conversations', methods=['DELETE'])
def clear_all_conversations():
    """
    Delete all conversations
    Returns:
        JSON: success message

    """
    try:
        num_deleted = Conversation.query.delete()
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f"All Conversations clered Successfuly ({num_deleted} deleted)"
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 200

@chat_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Get information about the AI model

    Returns:
        JSON: Model information
    """
    return jsonify({
        'status': 'success',
        'model_info': ai_model.get_mode_info()
    }), 200

def build_conversation_history(conversation_id,max_messages=20):
    """
    Build Conversation hisotry for AI
    Args:
        conversation_id(str): Conversation ID
        max_messages(int): Max number of messages to show
    :return:
        List: List of messages dictions for AI
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, Friendly and knowledgeable AI Assistant. Provide clear, accurate and helful responses.",
        }
    ]

    recent_messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp.desc()).limit(max_messages).all()

    recent_messages.reverse()

    for msg in recent_messages:
        messages.append({
            "role":msg.role,
            "content":msg.content,
        })

    return messages