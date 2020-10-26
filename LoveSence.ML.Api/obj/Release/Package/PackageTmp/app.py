"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
https://dev.to/paurakhsharma/flask-rest-api-part-0-setup-basic-crud-api-4650
"""

from flask import Flask, jsonify, request, Response
import MessageVerification as mv
app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

@app.route('/')
def index():
    return "I am up and running!"

@app.route('/api/message/Verify', methods=['POST'])
def Verify_message():
    message = request.get_json()
    retour = mv.gerer_verification(request, message)
    return retour, 200

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
