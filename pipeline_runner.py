from pipeline import _create_pipeline
from tfx import v1 as tfx
import os
PIPELINE_DIR = "/home/deepak/pipeline"

PIPELINE_NAME = "dibetes_prediction_pipeline"

PIPELINE_ROOT = os.path.join(PIPELINE_DIR,'pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_DIR,'metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join(PIPELINE_DIR,'serving_model', PIPELINE_NAME)
_trainer_module_file = "prediction.py"
DATA_ROOT = "processed_data"
tfx.orchestration.LocalDagRunner().run(
  _create_pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      data_root=DATA_ROOT,
      module_file=_trainer_module_file,
      serving_model_dir=SERVING_MODEL_DIR,
      metadata_path=METADATA_PATH))