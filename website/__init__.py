from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] ='qwerty'
    
    # Enable CORS for the entire app so frontend can hit the API
    CORS(app)
    
    from .views import views 
    
    app.register_blueprint(views,url_prefix= '/')
    
    return app

create_app()