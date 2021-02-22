"""Routes for parent Flask app."""
import os
from flask import render_template, url_for, redirect
from flask import current_app as app


@app.route("/", methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])
def home():
    """Redirects to forecasting page"""
    return redirect('/forecasting/')

@app.route("/methodology")
@app.route("/methodology/")
def method_page():
    """Rendering to methodology page"""
    return render_template('methodology.html')



@app.context_processor
def override_url_for():
    ''' New CSS is not udpated due to browser cache. This function appends time to css path to make the updated CSS seem
    like a new file. '''
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    ''' This function appends the date to the css path'''
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)