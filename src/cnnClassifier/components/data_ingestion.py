import os
import urllib.request as request
import zipfile
from src.cnnClassifier import logger
from src.cnnClassifier.utils.common import get_size
from src.cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        # ✅ Skip download if no URL provided — you placed the dataset manually
        if not self.config.source_URL:
            logger.info("No source URL — using locally placed dataset. Skipping download.")
            return

        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with info:\n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Unzips downloaded file into unzip_dir.
        Skipped automatically when no zip file is present (manual dataset case).
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info("No zip file found — skipping extraction. Dataset folder already in place.")
            return

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip to: {unzip_path}")