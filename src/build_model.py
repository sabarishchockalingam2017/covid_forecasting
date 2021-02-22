
import pandas as pd
from pmdarima import auto_arima
import numpy as np
from itertools import product
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging.config
from statsmodels.tools import eval_measures
from src.helpers import model_helpers

'''Functions to build forecasting models.'''

logger = logging.getLogger("build_model")
prediction_horizon = 15

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
    test_horizon = len(testdf)
    model = auto_arima(traindf)
    country_name = traindf.columns[0]
    preddates = pd.date_range(traindf.index[-1], periods=test_horizon+1, freq=traindf.index.freq)[1:]
    pred = pd.Series(model.predict(n_periods=test_horizon), index=preddates)
    pyhat = pd.Series(traindf.tail(1).squeeze(), index=[traindf.index[-1]]).append(pred)
    pyhat.rename_axis('Date', inplace=True)
    pyhat.rename(country_name, inplace=True)

    # updating model with test data to forecast unobserved dates
    model.update(testdf)
    forecastdates = pd.date_range(testdf.index[-1], periods=prediction_horizon+1, freq=testdf.index.freq)[1:]
    forecast = pd.Series(model.predict(n_periods=prediction_horizon), index=forecastdates)
    fyhat = pd.Series(testdf.tail(1).squeeze(),index=[testdf.index[-1]]).append(forecast)
    fyhat.rename_axis('Date', inplace=True)
    fyhat.rename(country_name, inplace=True)

    # merge for plotting
    tsdf = pd.merge(pyhat, fyhat, how='outer', on='Date')
    tsdf.columns = ['ARIMA - Test', 'ARIMA - Forecast']

    # calculating evaluations metrics by comparing to testdata
    test_rmse = eval_measures.rmse(testdf, pyhat, axis=None)
    test_mape = model_helpers.mape(testdf.iloc[:,0], pyhat.iloc[1:])
    arima_metrics = {'rmse':test_rmse, 'mape':test_mape}

    logger.info("ARIMA model built for {}.".format(country_name))
    return tsdf, arima_metrics

def hw_prediction(traindf, testdf):
    """ Builds many Holt-Winters/Exponential Smoothing models and returns best result
    Args:
        traindf(Series): Time series used to train Exponential Smoothing model from ExponentialSmoothing function.
        testdf(Series): Test data in continuation of traindf used compared with forecast to evaluate accuracy.
    Output:
        tsdf(DataFrame): Dataframe with dates as index and training data, test data and forecast as columns.
                        to conveniently plot with Dash.
    """

    country_name = traindf.columns[0]

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
    pyhat.rename('Holt-Winters - Test', inplace=True)

    fullseries = pd.concat([traindf, testdf.iloc[1:]])#.asfreq(traindf.index.freq)

    # refitting to forecast unobserved dates
    hw_latest = ExponentialSmoothing(fullseries,
                                     seasonal=hw_best.model.seasonal,
                                     trend=hw_best.model.trend,
                                     seasonal_periods=hw_best.model.seasonal_periods).fit()

    forecast = hw_latest.predict(start=len(fullseries), end=len(fullseries)+prediction_horizon)

    # adding last test data point to show continuation in plots
    fyhat = pd.Series(testdf.tail(1).squeeze(), index=[testdf.index[-1]]).append(forecast)
    fyhat.rename_axis('Date', inplace=True)
    fyhat.rename('Holt-Winters - Forecast', inplace=True)

    # merge for plotting
    tsdf = pd.merge(pyhat, fyhat, how='outer', on='Date')

    # calculating evaluations metrics by comparing to testdata
    test_rmse = eval_measures.rmse(testdf, pyhat, axis=None)
    test_mape = model_helpers.mape(testdf.iloc[:,0], pyhat.iloc[1:])
    hw_metrics = {'rmse':test_rmse, 'mape':test_mape}

    logger.info("Holt-Winter model built for {}.".format(country_name))

    return tsdf, hw_metrics
