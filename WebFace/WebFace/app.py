from flask import Flask
from WebFace.Views.index import bp as index_bp

app = Flask(__name__)

app.register_blueprint(index_bp)