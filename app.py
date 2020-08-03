#!flask/bin/python

from flask import Flask
from flask_cors import CORS
from apis import api

app = Flask(__name__)
#cors = CORS(app, resources={r"/foo": {"origins": "http://***REMOVED***"}})
#cors = CORS(app)
api.init_app(app)


if __name__=='__main__':
	#app.run(debug=True, port=5001, host='0.0.0.0')
        app.run()
	#app.run(debug=True, port=5001)
