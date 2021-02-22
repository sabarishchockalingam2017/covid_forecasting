import numpy as np
'''Functions to help when building models'''

def mape(actdata, preddata):
    '''Calculates Mean Absolute Percentage error given actual data and predicted data'''

    # replace values less than or equal to 1 with 1 so percentages are not exaggerated
    actdata[actdata <= 1] = 1
    preddata[preddata <= 1] = 1

    y_true = np.asarray(actdata)
    y_pred = np.asarray(preddata)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

