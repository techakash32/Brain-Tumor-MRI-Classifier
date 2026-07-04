from src.cnnClassifier import logger

from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline


# ─────────────────────────────────────────────
# Stage 1 — Data Ingestion
# ─────────────────────────────────────────────
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ─────────────────────────────────────────────
# Stage 2 — Prepare Base Model (VGG16 + 3-class head)
# ─────────────────────────────────────────────
STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ─────────────────────────────────────────────
# Stage 3 — Training
# ─────────────────────────────────────────────
STAGE_NAME = "Training"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ─────────────────────────────────────────────
# Stage 4 — Evaluation
# ─────────────────────────────────────────────
STAGE_NAME = "Evaluation"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    model_eval = EvaluationPipeline()
    model_eval.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e