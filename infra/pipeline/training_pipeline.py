"""
SageMaker Pipeline that trains the sepsis model from the raw data in S3.

    Prepare ──► Train ──► Evaluate ──► QualityGate ──(pass)──► Register
                                                   └─(fail)──► QualityGateFailed

Train runs either as a Processing job or as a Managed Spot Training job,
depending on TRAINING_MODE below.

Run from infra/ with the virtualenv active:

    python -m pipeline.training_pipeline show     # write the pipeline JSON to build/
    python -m pipeline.training_pipeline upsert   # create/update it in AWS
    python -m pipeline.training_pipeline start    # start a run

The long-lived pieces (buckets, the SageMaker role) come from the
SepsisFoundation CDK stack; this script only defines the workflow.
"""

import argparse
import atexit
import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import boto3
from sagemaker.core.image_uris import retrieve
from sagemaker.core.model_registry import create_model_package_from_containers
from sagemaker.core.processing import (FrameworkProcessor, PipelineSession,
                                       ProcessingInput, ProcessingOutput,
                                       ProcessingS3Input)
from sagemaker.core.shapes import ProcessingS3Output
from sagemaker.core.workflow import (ConditionGreaterThanOrEqualTo,
                                     ExecutionVariables, Join, JsonGet,
                                     ParameterFloat, ParameterString,
                                     PropertyFile)
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import (CacheConfig, ProcessingStep,
                                            TrainingStep)
from sagemaker.train import ModelTrainer
from sagemaker.train import defaults as train_defaults
from sagemaker.train.configs import (Compute, InputData, MetricDefinition,
                                     OutputDataConfig, SourceCode,
                                     StoppingCondition)

# SDK v3 pre-checks the training role on this machine against a generic
# list of actions, and blocks on VPC networking permissions
# (ec2:*NetworkInterface*, only needed for jobs run inside a VPC, which ours
# aren't) and on cloudwatch:PutMetricData (simulated without the namespace
# condition our role grants it under). The role is defined least-privilege
# in the CDK stack, and AWS still enforces its real permissions on every
# job, so use it as given instead of granting unused permissions.
train_defaults.TrainDefaults.get_role = staticmethod(
    lambda role=None, sagemaker_session=None: role)

if sys.platform == 'win32':
    # SDK bug on Windows: FrameworkProcessor deletes its temporary
    # sourcedir.tar.gz while the file is still open, which only Linux/macOS
    # allow. Give that one module an os.unlink that defers the delete to
    # interpreter exit instead. Not needed where CI runs (Linux).
    import sagemaker.core.processing as sm_processing

    _deferred: list[str] = []

    class _WindowsSafeOs(types.ModuleType):
        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def unlink(path):
            try:
                os.unlink(path)
            except PermissionError:
                _deferred.append(path)

    sm_processing.os = _WindowsSafeOs('os')
    atexit.register(lambda: [os.remove(p) for p in _deferred if os.path.exists(p)])

REGION = 'ap-south-1'
STACK_NAME = 'SepsisFoundation'
PIPELINE_NAME = 'sepsis-training'
MODEL_PACKAGE_GROUP = 'sepsis-warning'   # created by the CDK stack (model_registry flag)

# One environment for every step: AWS's prebuilt PyTorch CPU container
# (Python 3.11, like local development), plus container_requirements.txt.
PYTORCH_VERSION = '2.5'
PY_VERSION = 'py311'

# How the Train step runs:
#   'training-job' - a SageMaker Training job on Managed Spot capacity, with
#                    per-epoch metrics in the console (needs the spot training
#                    quota, L-4CEE6BA6).
#   'processing'   - inside a Processing job. Fallback for accounts with
#                    processing quota but no training-job quota.
TRAINING_MODE = 'training-job'

PROCESSING_INSTANCE = 'ml.t3.xlarge'   # 4 vCPU / 16 GB, burstable; has quota
TRAINING_INSTANCE = 'ml.m5.xlarge'     # 4 vCPU / 16 GB; for 'training-job' mode

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CODE_BUNDLE = HERE.parent / 'build' / 'sm_code'


def build_code_bundle() -> Path:
    """
    Stage exactly what the containers need: the sepsis package, the
    entry scripts and requirements.txt. SageMaker uploads this folder to
    S3 and unpacks it inside each job.
    """
    shutil.rmtree(CODE_BUNDLE, ignore_errors=True)
    shutil.copytree(REPO_ROOT / 'src' / 'sepsis', CODE_BUNDLE / 'sepsis',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for script in (HERE / 'entrypoints').glob('*.py'):
        shutil.copy(script, CODE_BUNDLE)
    shutil.copy(HERE / 'container_requirements.txt', CODE_BUNDLE / 'requirements.txt')
    return CODE_BUNDLE


def bundle_hash(folder: Path) -> str:
    """Hash of every file in the code bundle (paths + contents)."""
    digest = hashlib.sha256()
    for f in sorted(p for p in folder.rglob('*') if p.is_file()):
        digest.update(f.relative_to(folder).as_posix().encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()[:12]


def stack_outputs(boto_session: boto3.Session) -> dict[str, str]:
    stack = boto_session.client('cloudformation').describe_stacks(
        StackName=STACK_NAME)['Stacks'][0]
    return {o['OutputKey']: o['OutputValue'] for o in stack['Outputs']}


def build_pipeline(boto_session: boto3.Session) -> Pipeline:
    out = stack_outputs(boto_session)
    role = out['SageMakerRoleArn']
    data_bucket = out['DataBucketName']
    artifacts_bucket = out['ArtifactsBucketName']
    code_dir = build_code_bundle()
    # The code is uploaded under this name. Deriving it from the bundle's
    # contents (instead of the default timestamp) keeps step arguments
    # identical until the code changes, which is what step caching compares.
    code_version = bundle_hash(code_dir)
    code_dir = str(code_dir)

    # PipelineSession records each job's arguments instead of starting it,
    # which is how the SDK turns ordinary job code into pipeline steps.
    # Pointing it at our artifacts bucket stops the SDK creating its own
    # "sagemaker-<region>-<account>" bucket outside the CDK stack.
    session = PipelineSession(boto_session=boto_session,
                              default_bucket=artifacts_bucket,
                              default_bucket_prefix='pipeline')

    image = retrieve('pytorch', region=REGION, version=PYTORCH_VERSION,
                     py_version=PY_VERSION, instance_type=PROCESSING_INSTANCE,
                     image_scope='training')
    inference_image = retrieve('pytorch', region=REGION, version=PYTORCH_VERSION,
                               py_version=PY_VERSION, instance_type=PROCESSING_INSTANCE,
                               image_scope='inference')

    # ── Parameters: values you can override per run without redeploying ──
    raw_data = ParameterString('RawDataS3Uri', default_value=f's3://{data_bucket}/raw/')
    min_utility = ParameterFloat('MinTestUtility', default_value=0.20)

    # ── 1. Prepare ────────────────────────────────────────────────────────
    prepare = FrameworkProcessor(
        image_uri=image, role=role, instance_count=1, instance_type=PROCESSING_INSTANCE,
        command=['python3'], base_job_name='sepsis-prepare',
        max_runtime_in_seconds=3600, sagemaker_session=session,
    )
    step_prepare = ProcessingStep(
        name='Prepare',
        description='Features, labels and the patient-level train/val/test split',
        step_args=prepare.run(
            code='run_prepare.py', source_dir=code_dir, requirements='requirements.txt',
            job_name=f'sepsis-prepare-{code_version}',
            inputs=[ProcessingInput(input_name='raw', s3_input=ProcessingS3Input(
                s3_uri=raw_data, local_path='/opt/ml/processing/input/raw',
                s3_data_type='S3Prefix', s3_input_mode='File'))],
            # A fixed output path (no run ID) is what lets caching reuse it
            outputs=[ProcessingOutput(output_name='splits', s3_output=ProcessingS3Output(
                s3_uri=f's3://{data_bucket}/processed/',
                local_path='/opt/ml/processing/output', s3_upload_mode='EndOfJob'))],
        ),
        # Same code + same raw data -> reuse the last result instead of re-running
        cache_config=CacheConfig(enable_caching=True, expire_after='P30D'),
    )
    splits_uri = step_prepare.properties.ProcessingOutputConfig.Outputs['splits'].S3Output.S3Uri

    # ── 2. Train ──────────────────────────────────────────────────────────
    train_step = (train_as_processing_job if TRAINING_MODE == 'processing'
                  else train_as_spot_training_job)
    step_train, model_uri = train_step(
        session=session, role=role, image=image, code_dir=code_dir,
        code_version=code_version, splits_uri=splits_uri,
        artifacts_bucket=artifacts_bucket)

    # ── 3. Evaluate on the held-out test set ─────────────────────────────
    evaluate = FrameworkProcessor(
        image_uri=image, role=role, instance_count=1, instance_type=PROCESSING_INSTANCE,
        command=['python3'], base_job_name='sepsis-evaluate',
        max_runtime_in_seconds=1800, sagemaker_session=session,
    )
    report = PropertyFile(name='EvaluationReport', output_name='evaluation',
                          path='evaluation.json')
    step_evaluate = ProcessingStep(
        name='Evaluate',
        description='Score the once-only test set (PhysioNet Set B)',
        step_args=evaluate.run(
            code='run_evaluate.py', source_dir=code_dir, requirements='requirements.txt',
            job_name=f'sepsis-evaluate-{code_version}',
            inputs=[
                ProcessingInput(input_name='model', s3_input=ProcessingS3Input(
                    s3_uri=model_uri,
                    local_path='/opt/ml/processing/model',
                    s3_data_type='S3Prefix', s3_input_mode='File')),
                ProcessingInput(input_name='test', s3_input=ProcessingS3Input(
                    s3_uri=splits_uri, local_path='/opt/ml/processing/test',
                    s3_data_type='S3Prefix', s3_input_mode='File')),
            ],
            outputs=[ProcessingOutput(output_name='evaluation', s3_output=ProcessingS3Output(
                s3_uri=Join(on='/', values=[f's3://{artifacts_bucket}/evaluation',
                                            ExecutionVariables.PIPELINE_EXECUTION_ID]),
                local_path='/opt/ml/processing/evaluation', s3_upload_mode='EndOfJob'))],
        ),
        # Exposes evaluation.json to later steps, so conditions can read it
        property_files=[report],
    )

    # ── 5. Register (runs only if the quality gate passes) ────────────────
    # PipelineSession captures the CreateModelPackage request instead of
    # sending it; ModelStep turns the captured request into a pipeline step.
    # Every passing run becomes a new version in the Model Registry, marked
    # PendingManualApproval: a person reviews the metrics and approves it
    # before it can be deployed.
    session.init_model_step_arguments(types.SimpleNamespace(sagemaker_session=session))
    evaluation_json = Join(on='/', values=[
        step_evaluate.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri,
        'evaluation.json'])
    step_register = ModelStep(
        name='Register',
        description='Record the model and its test metrics as a new pending version',
        step_args=create_model_package_from_containers(
            session,
            containers=[{'Image': inference_image, 'ModelDataUrl': model_uri}],
            content_types=['application/json'],
            response_types=['application/json'],
            model_package_group_name=MODEL_PACKAGE_GROUP,
            # Shown on the version's page in the registry (CreateModelPackage format)
            model_metrics={'ModelQuality': {'Statistics': {
                'ContentType': 'application/json', 'S3Uri': evaluation_json}}},
            approval_status='PendingManualApproval',
            description='Sepsis early warning LSTM trained by the sepsis-training pipeline',
            customer_metadata_properties={
                'training_mode': TRAINING_MODE,
                'code_version': code_version,
                'pipeline_execution': ExecutionVariables.PIPELINE_EXECUTION_ID,
            },
        ),
    )

    # ── 4. Quality gate ──────────────────────────────────────────────────
    step_gate = ConditionStep(
        name='QualityGate',
        description='Test utility must reach MinTestUtility',
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=step_evaluate.name, property_file=report,
                         json_path='test.utility'),
            right=min_utility)],
        if_steps=[step_register],
        else_steps=[FailStep(
            name='QualityGateFailed',
            error_message=Join(on=' ', values=[
                'Test utility is below the minimum of', min_utility]))],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[raw_data, min_utility],
        steps=[step_prepare, step_train, step_evaluate, step_gate],
        sagemaker_session=session,
    )


def train_as_processing_job(*, session, role, image, code_dir, code_version,
                            splits_uri, artifacts_bucket):
    """
    Train inside a Processing job: run_train.py runs sepsis.train and packs
    the result as model.tar.gz. No Spot discount and no per-epoch metric
    charts, but it only needs processing quota.
    """
    trainer = FrameworkProcessor(
        image_uri=image, role=role, instance_count=1,
        instance_type=PROCESSING_INSTANCE, command=['python3'],
        base_job_name='sepsis-train', max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )
    step = ProcessingStep(
        name='Train',
        description='LSTM training (in a Processing job); epoch and thresholds chosen on validation',
        step_args=trainer.run(
            code='run_train.py', source_dir=code_dir, requirements='requirements.txt',
            job_name=f'sepsis-train-{code_version}',
            inputs=[ProcessingInput(input_name='splits', s3_input=ProcessingS3Input(
                s3_uri=splits_uri, local_path='/opt/ml/processing/input/data',
                s3_data_type='S3Prefix', s3_input_mode='File'))],
            outputs=[ProcessingOutput(output_name='model', s3_output=ProcessingS3Output(
                s3_uri=Join(on='/', values=[f's3://{artifacts_bucket}/models',
                                            ExecutionVariables.PIPELINE_EXECUTION_ID]),
                local_path='/opt/ml/processing/output', s3_upload_mode='EndOfJob'))],
        ),
    )
    model_dir = step.properties.ProcessingOutputConfig.Outputs['model'].S3Output.S3Uri
    return step, Join(on='/', values=[model_dir, 'model.tar.gz'])


def train_as_spot_training_job(*, session, role, image, code_dir, code_version,
                               splits_uri, artifacts_bucket):
    """
    Train as a SageMaker Training job on Managed Spot capacity (needs the
    spot training quota). SageMaker packs /opt/ml/model into model.tar.gz
    and charts the scraped metrics per epoch.
    """
    trainer = ModelTrainer(
        sagemaker_session=session, role=role, training_image=image,
        base_job_name='sepsis-train',
        source_code=SourceCode(
            source_dir=code_dir, requirements='requirements.txt',
            # Channels are always mounted at /opt/ml/input/data/<name>, and
            # whatever is written to /opt/ml/model becomes model.tar.gz.
            command='python -m sepsis.train '
                    '--data-dir /opt/ml/input/data/train --model-dir /opt/ml/model'),
        compute=Compute(instance_type=TRAINING_INSTANCE, instance_count=1,
                        volume_size_in_gb=10, enable_managed_spot_training=True),
        # Spot: run at most 1 h, and wait at most 2 h in total for spare capacity
        stopping_condition=StoppingCondition(max_runtime_in_seconds=3600,
                                             max_wait_time_in_seconds=7200),
        output_data_config=OutputDataConfig(s3_output_path=f's3://{artifacts_bucket}/models'),
    ).with_metric_definitions([
        # Scraped from the "key=value" lines sepsis.train prints each epoch
        MetricDefinition(name='val_utility', regex=r'val_utility=([-0-9.]+)'),
        MetricDefinition(name='val_auroc', regex=r'val_auroc=([-0-9.]+)'),
        MetricDefinition(name='train_loss', regex=r'train_loss=([-0-9.]+)'),
    ])
    step = TrainingStep(
        name='Train',
        description='LSTM on Spot capacity; epoch and thresholds chosen on validation',
        step_args=trainer.train(input_data_config=[
            InputData(channel_name='train', data_source=splits_uri)]),
    )
    return step, step.properties.ModelArtifacts.S3ModelArtifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('action', choices=['show', 'upsert', 'start'])
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=REGION)
    pipeline = build_pipeline(boto_session)

    if args.action == 'show':
        path = CODE_BUNDLE.parent / 'pipeline_definition.json'
        path.write_text(json.dumps(json.loads(pipeline.definition()), indent=2))
        print(f'Pipeline definition written to {path}')
    elif args.action == 'upsert':
        role = stack_outputs(boto_session)['SageMakerRoleArn']
        response = pipeline.upsert(
            role_arn=role, description='Sepsis early warning: prepare, train, evaluate')
        print(f"Pipeline ready: {response['PipelineArn']}")
    else:
        execution = pipeline.start()
        print(f'Started: {execution.arn}')


if __name__ == '__main__':
    main()
