import os
import kagglehub
import shutil
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import *
import zipfile

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, dataset_name:str, target_dir:str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self):
        raw_dir = os.path.join(self.target_dir, "raw")
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir)
                logger.info(f"Created {raw_dir}")
            except Exception as e:
                logger.error("Error in creating raw directory")
                raise CustomException("Failed to create raw dir.", e)
        return raw_dir
    
    def extract_images_and_labels(self, path:str, raw_dir:str):
        try:
            logger.info("Extracting images and labels zip file")
            if path.endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(path)
            images_folder = os.path.join(path,"Images")
            labels_folder = os.path.join(path,"Labels")

            if os.path.exists(images_folder):
                shutil.move(images_folder, os.path.join(raw_dir, "Images"))
                logger.info("Images moved successfully...")
            else:
                logger.info("Images folder doesn't exist")

            if os.path.exists(labels_folder):
                shutil.move(labels_folder, os.path.join(raw_dir, "Labels"))
                logger.info("Labels moved successfully...")
            else:
                logger.info("Lables folder doesn't exist")
        except Exception as e:
            logger.error("Error in extracting images and labels")
            raise CustomException("Failed to extract images and labels.", e)
    
    def download_dataset(self, raw_dir:str):
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Downloaded data from {path}")
            self.extract_images_and_labels(path,raw_dir)
        except Exception as e:
            logger.error("Error in downloading dataset")
            raise CustomException("Failed to download dataset.", e)
        
    def run(self):
        try:
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
        except Exception as e:
            logger.error("Error while data ingestion pipeline")
            raise CustomException("Data ingestion pipeline failed.", e)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion(DATASET_NAME, TARGET_DIR)
    data_ingestion.run()
        





    
