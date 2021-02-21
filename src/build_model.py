'''Functions to build forecasting models.'''
import pandas as pd
from pmdarima import auto_arima
import numpy as np
from itertools import product
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def arima_prediction(traindf, testdf):
    """ Uses auto_arima to find order and build best ARIMA model
    Args:
        traindf(Series): Time series used to train ARIMA model from auto_arima function.
        testdf(Series): Test data in continuation of traindf used compared with ARIMA forecast to evaluate accuracy.

    Output:
        tsdf(DataFrame): Dataframe with dates as index and training data, test data and forecast as columns.
                        to conveniently plot with Dash.

    """
    # make predictions
    test_horizon = len(testdf) - 1
    model = auto_arima(traindf)
    preddates = pd.date_range(traindf.index[-1], periods=test_horizon + 1, freq=traindf.index.freq)[1:]
    pred = pd.Series(model.predict(n_periods=test_horizon), index=preddates)
    pyhat = pd.Series(traindf.tail(1).squeeze(), index=[traindf.index[-1]]).append(pred)
    pyhat.rename_axis('Date', inplace=True)
    pyhat.rename(traindf.columns[0], inplace=True)

    # merge for plotting
    tsdf = pd.merge(traindf, testdf, how='outer', on='Date')
    tsdf = pd.merge(tsdf, pyhat, how='outer', on='Date')
    tsdf.columns = ['Training Data', 'Test Data', 'ARIMA - Forecast']

    return tsdf

def hw_prediction(traindf, testdf):
    """ Builds many Holt-Winters/Exponential Smoothing models and returns best result
    Args:
        traindf(Series): Time series used to train Exponential Smoothing model from ExponentialSmoothing function.
        testdf(Series): Test data in continuation of traindf used compared with forecast to evaluate accuracy.
    Output:
        tsdf(DataFrame): Dataframe with dates as index and training data, test data and forecast as columns.
                        to conveniently plot with Dash.
    """

    # multiplicative fitting does not work with zero or negative values
    if np.any(traindf<=0) or np.any(testdf<=0):
        hw_params = [[None, 'additive'], [None, 'additive'], [7, 15, 30, 90]]
    else:
        hw_params = [[None, 'additive', 'multiplicative'], [None, 'additive', 'multiplicative'], [7, 15, 30, 90]]

    # creating of grid of seasonal, multiplicative and seasonal periods parameters to find best combination
    hw_par_iter = list(product(*hw_params))
    # fitting Holt-Winters/Exponential Smoothing models
    hw_models = [ExponentialSmoothing(traindf, seasonal=s, trend=t, seasonal_periods=p).fit() for s, t, p in
                hw_par_iter]
    hw_best = sorted(hw_models, key=lambda x: x.aic)[0]
    # predictions from best model
    pred = hw_best.predict(start=len(traindf), end=len(traindf) + len(testdf) - 1)
    # adding last point in train data to make plots continous
    pyhat = pd.Series(traindf.tail(1).squeeze(), index=[traindf.index[-1]]).append(pred)
    pyhat.rename_axis('Date', inplace=True)
    pyhat.rename('Holt-Winters - Forecast', inplace=True)

    return pyhat
