"""
SageMaker Processing entry point for the Prepare step.

SageMaker mounts the raw S3 data at /opt/ml/processing/input/raw and
uploads whatever is left in /opt/ml/processing/output to S3 when the job
ends. Those are sepsis.prepare's default paths, so it runs unchanged.
"""

import runpy

runpy.run_module('sepsis.prepare', run_name='__main__')
