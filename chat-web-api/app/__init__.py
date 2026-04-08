from multiprocessing.reduction import register

from flask import Flask
from flask_cors import CORS
from config import config
from app.models.database import db

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    db.init_app(app)

    CORS(app,origins=app.config['CORS_ORIGINS'])

    from app.controllers.chat_controller import chat_bp
    app.register_blueprint(chat_bp,url_prefix='/api')

    with app.app_context():
        db.create_all()

    register_error_handlers(app)

    return app

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(error):
        return {
            'success': False,
            'error':'Resource Not Found',
            'status':404
        },404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return {
            'success': False,
            'error':'Method Not Allowed',
            'status':405
        },405

    @app.errorhandler(500)
    def internal_server_error(error):
        return {
            'success': False,
            'error':'Internal Server Error',
            'status':500
        },500
    @app.errorhandler(Exception)
    def handle_exception(error):
        return {
            'success': False,
            'error':'An Unexpected Internal Server Error Occured',
            'details':str(error),
            'status':500
        },500