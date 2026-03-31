"""Helper Fucntions"""
import  re
from functools import wraps
from flask import request, jsonify

def validate_input(required_fields):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            if not data:
                return jsonify({'message': 'No data provided'}), 400

            missing_fields=[]
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)

            if missing_fields:
                return jsonify({'message': f'Missing {", ".join(missing_fields)}'}), 400
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_email(email):
    pattern = r'A[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_input(text):
    text = re.sub(r'<[^>]+>','',text)
    text = ' '.json(text.split())
    return text.strip()

def format_response(data,status='success',message=None):
    response = {
         'status':status,
        'data':data,
    }

    if message:
        response['message'] = message

    return message
