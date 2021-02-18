'''Download data from kaggle which is curated from JHU github data'''
import os
import kaggle
from config.main_config import DATA_PATH

kaggle.api.authenticate()

kaggle.api.dataset_download_files('antgoldbloom/covid19-data-from-john-hopkins-university', path=DATA_PATH, unzip=True)
