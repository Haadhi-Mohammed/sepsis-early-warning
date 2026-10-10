"""
Prepare an approved model version for serving.

Takes an Approved version from the SageMaker Model Registry, adds the
serving code (code/inference.py, the sepsis package, requirements.txt) to
its model.tar.gz, uploads the result to
s3://<artifacts bucket>/serving/v<version>-<code hash>/model.tar.gz, and
records the version and code hash in cdk.json, so the next
`npx cdk deploy SepsisServing` serves it.

Run from infra/ with the virtualenv active:
    python -m serving.package_model            # latest approved version
    python -m serving.package_model --version 1
"""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import boto3

REGION = 'ap-south-1'
STACK_NAME = 'SepsisFoundation'
MODEL_PACKAGE_GROUP = 'sepsis-warning'

HERE = Path(__file__).resolve().parent
INFRA = HERE.parent
REPO_ROOT = INFRA.parent
BUILD = INFRA / 'build' / 'serving'


def code_files() -> dict[str, Path]:
    """Archive path -> local file for everything that goes into code/."""
    files = {'code/inference.py': HERE / 'inference.py',
             'code/requirements.txt': HERE / 'requirements.txt'}
    pkg = REPO_ROOT / 'src' / 'sepsis'
    for f in sorted(pkg.rglob('*.py')):
        files[f'code/sepsis/{f.relative_to(pkg).as_posix()}'] = f
    return files


def code_hash(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for arcname, path in sorted(files.items()):
        digest.update(arcname.encode())
        # Line endings normalised: CRLF (Windows checkout) and LF hash alike
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()[:12]


def approved_version(sm, version: int | None) -> dict:
    if version is None:
        summaries = sm.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP, ModelApprovalStatus='Approved',
            SortBy='CreationTime', SortOrder='Descending', MaxResults=1,
        )['ModelPackageSummaryList']
        if not summaries:
            raise SystemExit(f'No Approved versions in {MODEL_PACKAGE_GROUP}')
        arn = summaries[0]['ModelPackageArn']
    else:
        arn = sm.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP, MaxResults=100,
        )['ModelPackageSummaryList']
        arn = next((p['ModelPackageArn'] for p in arn
                    if p['ModelPackageVersion'] == version), None)
        if arn is None:
            raise SystemExit(f'Version {version} not found in {MODEL_PACKAGE_GROUP}')
    package = sm.describe_model_package(ModelPackageName=arn)
    if package['ModelApprovalStatus'] != 'Approved':
        raise SystemExit(f"Version {package['ModelPackageVersion']} is "
                         f"{package['ModelApprovalStatus']}; approve it first")
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--version', type=int, help='registry version (default: latest approved)')
    args = parser.parse_args()

    session = boto3.Session(region_name=REGION)
    sm, s3 = session.client('sagemaker'), session.client('s3')
    stack = session.client('cloudformation').describe_stacks(StackName=STACK_NAME)['Stacks'][0]
    artifacts_bucket = next(o['OutputValue'] for o in stack['Outputs']
                            if o['OutputKey'] == 'ArtifactsBucketName')

    package = approved_version(sm, args.version)
    version = package['ModelPackageVersion']
    source = package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    files = code_files()
    code_version = code_hash(files)
    print(f'Version {version} (Approved), model {source}, serving code {code_version}')

    # Original model files plus code/, in a new archive
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True)
    bucket, key = source.removeprefix('s3://').split('/', 1)
    original = io.BytesIO(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    packed = BUILD / 'model.tar.gz'
    with tarfile.open(fileobj=original, mode='r:gz') as src, tarfile.open(packed, 'w:gz') as dst:
        for member in src.getmembers():
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        for arcname, path in files.items():
            dst.add(path, arcname=arcname)

    target_key = f'serving/v{version}-{code_version}/model.tar.gz'
    s3.upload_file(str(packed), artifacts_bucket, target_key)
    print(f'Uploaded s3://{artifacts_bucket}/{target_key}')

    cdk_json = INFRA / 'cdk.json'
    cfg = json.loads(cdk_json.read_text())
    cfg['context']['serving'] = {'model_version': version, 'code_version': code_version}
    cdk_json.write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'cdk.json: serving = version {version}, code {code_version}. '
          'Next: npx cdk diff SepsisServing, then npx cdk deploy SepsisServing')


if __name__ == '__main__':
    main()
