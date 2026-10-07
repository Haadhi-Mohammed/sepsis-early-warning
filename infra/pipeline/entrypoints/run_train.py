"""
SageMaker Processing entry point for the Train step.

Used while the account has no training-job quota (TRAINING_MODE =
'processing' in training_pipeline.py). A Processing job can run any script,
so it runs sepsis.train exactly as a Training job would, then packs the
bundle into model.tar.gz, the same layout a Training job produces, so the
Evaluate step and serving don't care which ran.
"""

import runpy
import sys
import tarfile
from pathlib import Path

DATA_DIR = '/opt/ml/processing/input/data'     # Prepare step's splits
BUNDLE_DIR = Path('/opt/ml/processing/bundle')  # scratch: unpacked bundle
OUT_DIR = Path('/opt/ml/processing/output')     # uploaded to S3 at job end

sys.argv = ['train', '--data-dir', DATA_DIR, '--model-dir', str(BUNDLE_DIR)]
runpy.run_module('sepsis.train', run_name='__main__')

OUT_DIR.mkdir(parents=True, exist_ok=True)
with tarfile.open(OUT_DIR / 'model.tar.gz', 'w:gz') as tar:
    for f in sorted(BUNDLE_DIR.iterdir()):
        tar.add(f, arcname=f.name)
print(f'Wrote {OUT_DIR / "model.tar.gz"}')
