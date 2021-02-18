from flask import Flask, render_template, url_for, flash
import logging
import logging.config
import os
import pandas as pd
from pathlib import Path
from flask_app import init_app

'''This is the flask flask_app and builds the webpage interface for the analysis.'''

app = init_app()


logger = logging.getLogger("flask_app")



if __name__ == "__main__":
    app.run(debug=app.config['DEBUG'], port=app.config['PORT'])
