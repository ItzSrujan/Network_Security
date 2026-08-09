import os, sys
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer

from network_security.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformConfig,
    ModelTrainerConfig
)

from network_security.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Start data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed {data_ingestion_artifact}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingest_artifact : DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config = self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact = data_ingest_artifact, data_validation_config = data_validation_config)
            logging.info("Initiate the data validation")
            data_valid_artifact = data_validation.initiate_data_validation()
            return data_valid_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact : DataValidationArtifact):
            try:
                data_transformation_config = DataTransformConfig(training_pipeline_config = self.training_pipeline_config)
                data_transformation = DataTransformation(data_validation_artifact = data_validation_artifact , data_transformation_config = data_transformation_config)
                logging.info("Initiate the data transformation")
                data_transform_artifact = data_transformation.initiate_data_validation()
                return data_transform_artifact
            except Exception as e:
                raise NetworkSecurityException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact : DataValidationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config : ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config = self.training_pipeline_config
            )
            
            model_trainer = ModelTrainer(
                data_transformation_artifact = data_transformation_artifact,
                model_trainer_config = self.model_trainer_config
             )
            
            logging.info("Initiate model training")
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingest_artifact = data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact = data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact = data_transformation_artifact)
            
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)