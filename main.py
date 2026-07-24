from network_security.components.data_ingestion import DataIngestion
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException
from network_security.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig

import sys

if __name__ == "__main__" :
    try:
        train_pipeline_config = TrainingPipelineConfig()
        data_ingest_config = DataIngestionConfig(train_pipeline_config)
        data_ingestion = DataIngestion(data_ingest_config)
        
        logging.info("Initiate the data ingestion")
        
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    
    except Exception as e:
        raise NetworkSecurityException(e,sys)