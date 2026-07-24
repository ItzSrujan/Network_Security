from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
## configuration of the data ingestion config
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import certifi
import pandas as pd
import numpy as np
from pymongo.mongo_client import MongoClient
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

ca = certifi.where()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def export_collection_as_dataframe(self):
        # read the data from mongodb
        try:
            db_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = MongoClient(MONGO_DB_URL, tlsCAFile = ca) 
            collection = self.mongo_client[db_name][collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop("_id", axis=1)
            
            df.replace({"na":np.nan}, inplace = True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e,sys)        
        
    def export_data_to_feature_store(self, dataframe : pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok = True)
            dataframe.to_csv(feature_store_file_path, index = False, header = True)
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
    def data_split(self, dataframe : pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, 
                test_size = self.data_ingestion_config.train_test_split_ratio
            )
            
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited data_split method of Data_Ingestion class")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(
                dir_path,
                exist_ok = True
            )
            
            logging.info("Exporting train and test file path")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index = False,
                header = True
            )
            
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index = False,
                header = True
            )
            
            logging.info("Exported train and test file path")
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_to_feature_store(dataframe)
            self.data_split(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(train_file_path = self.data_ingestion_config.training_file_path,
                                                            test_file_path = self.data_ingestion_config.testing_file_path
            )
            
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)