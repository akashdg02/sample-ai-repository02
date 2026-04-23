import uuid
import re
from datetime import datetime
from functools import wraps
from flask import request,jsonify

def generate_conversation_id():
    return f"conv{uuid.uuid4().hex[:12]}"

def validate_request(required_fields):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            if not data:
                return jsonify({
                    "status": False,
                    "error":"No Json data provided"
                }),400

            missing_fields = []
            for field in required_fields:
                if field not in data or not data[field]:
                    missing_fields.append(field)

            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': "Missing required fields",
                    'missing_fields': missing_fields
                }),400

            return f(*args, **kwargs)
        return wrapper
    return decorator

def sanitize_input(text):
    if not text:
        return ''

    text = ''.join(text.split())

    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'javascript:','',text,flags=re.IGNORECASE)

    return text.strip

def format_timestamp(dt):
    if not dt:
        return None

    return dt.isoformat()

def truncate_text(text,max_length=100):
    if not text:
        return ''

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."

def error_response(message,status_code=400,**kwargs):
    response = {
        'success': False,
        'error': message
    }

    response.update(kwargs)

    return jsonify(response),status_code

def success_response(data=None, message=None, status_code=200):
    response = {
        'success': True,
    }

    if data is not None:
        response['data']=data

    if message:
        response['message'] = message

    return jsonify(response),status_code


def get_client_ip():
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split('.')[0]
    return request.remote_addr

def parse_bool(value):
    if isinstance(value,bool):
        return value
    if isinstance(value,str):
        return value.lower() in ('true' ,'1','yes','on')

    return bool(value)