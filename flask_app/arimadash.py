from dash import Dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from bs4 import BeautifulSoup
from flask import render_template
import pandas as pd
from src.build_model import arima_prediction
import numpy as np

def create_dashboard(server, countrydf):
    """Create a Plotly Dash dashboard.
    Args:
    server(Flask server object): parent flask server object passed through to be used by Dash.
    countrydf(DataFrame): Data frame of covid time series data from world countries.

    Output:
    dash_app.server(Flask server object): Flask server now with Dash app included.
    """

    # splitting data to test and train
    # data starts with US as default
    dates = pd.to_datetime(countrydf['Country/Region'][1:], format="%m/%d/%y")
    tsdata = pd.DataFrame(countrydf['US'][1:])
    tsdata.index = dates
    tsdata.index.freq = 'd'  # setting time frequency to be by date
    tsdata = tsdata.rename_axis('Date')
    tsdata = tsdata[tsdata.index > '03-01-2020']

    # splitting 80% into trainset and 20% into testset
    sersplit = round(len(tsdata) * 0.8)
    traindata = tsdata.iloc[:sersplit]
    testdata = tsdata.iloc[sersplit:]

    # prediction horizon - how far to forecast
    pred_horizon = len(testdata)

    # making predictions
    preddf = arima_prediction(traindata, testdata)

    dash_app = Dash(
        server=server
    )

    # Create Dash Layout
    dash_app.layout = html.Div([
        html.H1("COVID-19 Forecasting - ARIMA Model"),
        dcc.Graph(id='graph'),
        html.Label([
            "Country",
            dcc.Dropdown(
                id='country-dropdown', clearable=False,
                value='US', options=[
                {'label': c[0] + c[1], 'value': c[0]}
                for c in zip(countrydf.columns[1:],
                             countrydf.iloc[0,1:].replace(np.nan,'').apply(lambda x:'' if x=='' else ' - '+x))
            ])
        ]),
    ])

    # route to arima dash
    @server.route('/arimadash/')
    def arimadashapp():
        soup = BeautifulSoup(dash_app.index(),'html.parser')
        footer = soup.footer
        return render_template('arimadash.html', title='ARIMA', footer=footer)

    # Define callback to update graph
    @dash_app.callback(
        Output('graph', 'figure'),
        [Input("country-dropdown", "value")]
    )
    def update_figure(country):
        # skipping first row (header row) and resetting index to begin at 0
        dates = pd.to_datetime(countrydf['Country/Region'][1:], format="%m/%d/%y")
        tsdata = pd.DataFrame(countrydf[country][1:])
        tsdata.index = dates
        tsdata.index.freq = 'd'  # setting time frequency to be by date
        tsdata = tsdata.rename_axis('Date')
        tsdata = tsdata[tsdata.index > '03-01-2020']

        # splitting 80% into trainset and 20% into testset
        sersplit = round(len(tsdata) * 0.8)
        traindata = tsdata.iloc[:sersplit]
        testdata = tsdata.iloc[sersplit - 1:]

        plotdf = arima_prediction(traindata, testdata)

        fig = px.line(plotdf, title='Confirmed Cases Forecasting - {country}'.format(country=country),
                      labels={'y': 'Confirmed Cases'})
        fig.update_layout(yaxis_title="Confirmed Cases",
                          legend_title=None)
        return fig
    return dash_app.server