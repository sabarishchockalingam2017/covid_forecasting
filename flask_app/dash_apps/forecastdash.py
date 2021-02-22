from dash import Dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from bs4 import BeautifulSoup
from flask import render_template
import pandas as pd
from src.build_model import arima_prediction, hw_prediction
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
                            html.H4("COVID-19 Forecasting"),
                            dcc.Loading(id="loading-graph",
                                        type="default",
                                        children=dcc.Graph(id='graph')
                                        ),
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
    @server.route('/forecasting/')
    def forecastdashapp():
        soup = BeautifulSoup(dash_app.index(),'html.parser')
        footer = soup.footer
        return render_template('forecasting.html', title='Forecasting Models', footer=footer)

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

        plotdf = pd.merge(traindata, testdata, how='outer', on='Date')
        plotdf.columns = ['Train Data', 'Test Data']

        # arima prediction
        arimadf, arimametrics = arima_prediction(traindata, testdata)
        plotdf = pd.merge(plotdf, arimadf, how='outer', on='Date')

        # holt-winters prediction
        hw_pred, hw_metrics = hw_prediction(traindata, testdata)
        plotdf = plotdf.merge(hw_pred, how='outer', on='Date')

        fig = px.line(plotdf, title='Confirmed Cases Forecasting - {country}'.format(country=country),
                      labels={'y': 'Confirmed Cases'})

        # adding end of observed data line
        fig.add_vline(x=tsdata.index[-1],
                      line_width=3,
                      line_dash='dash',
                      line_color='red')
        fig.add_vrect(x0=tsdata.index[-63], x1=tsdata.index[-1],
                      annotation_text="End of Observed Data",
                      annotation_position="top right",
                      line_width=0)

        # adding metrics
        metrics_text = "ARIMA RMSE: {}<br> ARIMA MAPE: {}<br> Holt-Winters RMSE: {}<br> Holt-Winters MAPE: {}"\
            .format(round(arimametrics['rmse'], 2), round(arimametrics['mape'], 2),
                    round(hw_metrics['rmse'], 2), round(hw_metrics['mape'], 2))

        fig.add_annotation(x=1.24,
                           y=0.2,
                           xref="paper",
                           yref="paper",
                           showarrow=False,
                           align="left",
                           text=metrics_text)

        fig.update_layout(yaxis_title="Confirmed Cases",
                          legend_title=None)
        return fig
    return dash_app.server