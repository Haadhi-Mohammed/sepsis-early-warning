"""
SageMaker Processing entry point for the Evaluate step.

The Train step's output arrives as model.tar.gz (SageMaker always
compresses /opt/ml/model), so unpack it before handing the folder to
sepsis.evaluate.
"""

import runpy
import sys
import tarfile
from pathlib import Path

MODEL_TAR = Path('/opt/ml/processing/model/model.tar.gz')
MODEL_DIR = Path('/opt/ml/processing/model_extracted')

MODEL_DIR.mkdir(parents=True, exist_ok=True)
with tarfile.open(MODEL_TAR) as tar:
    try:
        tar.extractall(MODEL_DIR, filter='data')   # refuses unsafe paths
    except TypeError:                              # Python < 3.11.4
        tar.extractall(MODEL_DIR)                  # our own training output

sys.argv = ['evaluate',
            '--model-dir', str(MODEL_DIR),
            '--data-dir', '/opt/ml/processing/test',
            '--out-dir', '/opt/ml/processing/evaluation']
runpy.run_module('sepsis.evaluate', run_name='__main__')
