from flask import Flask
from flask_cors import CORS
import logging
from .config import Config

# Initialisation de l'application
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, origins=["http://localhost:3000"], methods=['GET', 'POST'], allow_headers=['Content-Type'])

    # Configurer le logger
    logging.basicConfig(filename="logs/app.log", level=logging.INFO)

    # Enregistrer les blueprints (routes)
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app