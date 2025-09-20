import os
import pandas
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint

logger = get_logger(__name__)

class ModelTraining:
    """Class to train the model"""
    def __init__(self, train_path, test_path, model_ouput):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output = model_ouput

        self.param_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loadind data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info("Data splitted successfull for model training")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error while loading the data {e}")
            raise CustomException("Failed to load the data", e)
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing our model")
            



            




    
