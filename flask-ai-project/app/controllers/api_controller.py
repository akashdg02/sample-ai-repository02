"""
API Controller
"""

from flask import Blueprint, request, jsonify
from app.models.data_model import DataModel
from app.utils.helpers import validate_input

api_bp = Blueprint('api', __name__)

data_model = DataModel()

@api_bp.route('/health',methods=['GET'])
def health_check():
    return jsonify({
        'status':'healthy',
        'message':'API is running'
    }),200

@api_bp.route('/echo',methods=['POST'])
def echo():
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                'error':'Missing message field`'
            }),400

        message = data['message']

        return jsonify({
            'received':message,
            'length':len(message),
            'timestamp':data_model.get_timestamp()
        }), 200
    except Exception as e:
        return jsonify({
            'error':str(e)
        }),500

@api_bp.route('/process',methods=['POST'])
def process_data():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error':'Missing text field`'
            }),400

        text= data['text']
        options = data.get('options',{})
        result = data_model.process_text(text,options)

        return jsonify({
            'original':text,
            'processed':result,
            'options_applied':options,
        }),200
    except Exception as e:
        return jsonify({
            'error':str(e)
        }), 500

@api_bp.route('/calculate',methods=['POST'])
def calculate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error':'No Data Provide'
            }),400

        operation = data.get('operation')
        number = data.get('number')
        result = data_model.calculate(operation, number)
        return jsonify({
            'operation':operation,
            'number':number,
            'result':result
        }),200
    except ValueError as e:
        return jsonify({
            'error':f'Invalid Input: {str(e)}'
        })
    except Exception as e:
        return jsonify({
            'error':str(e)
        }),500