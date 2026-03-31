import os
from app import create_app

config_name = os.getenv('FLASK_ENV','development')

app=create_app(config_name)

if __name__ == '__main__':
    appHost=os.getenv('FLASK_HOST','0.0.0.0')
    appPort=os.getenv('FLASK_POST',5000)
    appDebug=os.getenv('FLASK_DEBUG',True)
    app.run(
        host=appHost,
        port=appPort,
        debug=appDebug
    )