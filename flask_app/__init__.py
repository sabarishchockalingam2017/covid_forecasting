"""Initialize Flask flask_app"""
from flask import Flask
from flask_app.dash_apps import forecastdash
import pandas as pd
from config.main_config import DATA_PATH
from pathlib import Path

def init_app():
    """Construct core Flask app."""
    flask_app = Flask(__name__, instance_relative_config=False)
    flask_app.config.from_object('config.flask_config')

    # loading data
    casesdf = pd.read_csv(Path(DATA_PATH,'CONVENIENT_global_confirmed_cases.csv'))

    with flask_app.app_context():
        # Import parts of core Flask flask_app
        from . import routes

        # creating dashboard for forecast modelling
        flask_app = forecastdash.create_dashboard(flask_app, casesdf)


        return flask_app