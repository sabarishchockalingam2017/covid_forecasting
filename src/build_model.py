'''Functions to build forecasting models.'''
import pandas as pd
from pmdarima import auto_arima

def arima_prediction(traindf, testdf):
    """ Uses auto_arima to find order and build best ARIMA model
    Args:
        traindf(Series): Time series used to train ARIMA model from auto_arima function.
        testdaf(Series): Test data in continuation of traindf used compared with ARIMA forecast accuracy.

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