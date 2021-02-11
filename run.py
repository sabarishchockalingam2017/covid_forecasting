import argparse
import logging.config
from config.main_config import LOGGING_PATH, LOGGING_CONFIG
from app.app import app

''' This is a central location to run all files.'''

logging.config.fileConfig(LOGGING_CONFIG,
                          disable_existing_loggers=False,
                          defaults={'log_dir': LOGGING_PATH})
logger = logging.getLogger("run_covid_forecasting")

def run_app():
    'Boots up app on server.'
    app.run(debug=app.config['DEBUG'], port=app.config['PORT'])

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Run app or source code")

    subparsers = parser.add_subparsers()

    args = parser.parse_args()

    run_app()