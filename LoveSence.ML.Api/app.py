
from flask import Flask, json, request, Response
import MessageVerification as mv
import CoprusService as Coprus_svc
import MLExperiencesService as ml_experiences_svc
import SessionVerificationService as session_verification_svc

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

@app.route('/api/corpus')
def obtenir_corpus():
    message = request.get_json()
    retour = Coprus_svc.obtenir_corpus(request)
    return Response(json.dumps(retour), mimetype="application/json", status=200)

@app.route('/api/mlexperience')
def obtenir_ml_experiences():
    message = request.get_json()
    retour = ml_experiences_svc.obtenir_experiences(request)
    return Response(json.dumps(retour), mimetype="application/json", status=200)

@app.route('/api/sessionverification')
def obtenir_sessions_verification():
    message = request.get_json()
    retour = session_verification_svc.obtenir_sessions(request)
    return Response(json.dumps(retour), mimetype="application/json", status=200)

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, port=None)
