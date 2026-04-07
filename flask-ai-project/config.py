"""
Application Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base Configuration"""

    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY')
    DEBUG = os.getenv('FLASK_DEBUG')

    #API Settings
    API_TIMEOUT = 30
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 #16MB max file size

    # CORS Settings
    CORS_HEADERS = 'Content-Type'

class DevelopmentConfig(Config):
    """Development Configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production Configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """ Testing Configuration"""
    DEBUG = True
    TESTING = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}