# Sagemaker
Aiming to use sagemaker without studio...

## Experiments

## Model registry
Like MLflow it can version/track model iterations.

Create a model package group:

```python
import boto3
sm_client = boto3.client('sagemaker')

response = sm_client.create_model_package_group(
    ModelPackageGroupName="my-models-group",
    ModelPackageGroupDescription="My ML models"
)
```
Register a model package:
```python
model_package_input = {
    "ModelPackageGroupName": "my-models-group",
    "ModelPackageDescription": "My model package",
    "InferenceSpecification": {
        "Containers": [{
            "Image": "<your-ecr-image-uri>",
            "ModelDataUrl": "s3://bucket/model.tar.gz"
        }],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    },
    "ModelApprovalStatus": "PendingManualApproval"
}
response = sm_client.create_model_package(**model_package_input)
```

List model packages in a group:
```python
response = sm_client.list_model_packages(
    ModelPackageGroupName="my-models-group"
)
```

Update approval status:
```python
response = sm_client.update_model_package(
    ModelPackageArn="<model-package-arn>",
    ModelApprovalStatus="Approved"
)
```

Get model package details:
```python
response = sm_client.describe_model_package(
    ModelPackageName="<model-package-arn>"
)
```

## Processing job
```python
import sagemaker.sklearn.processing

sklearn_processor = sagemaker.sklearn.processing.SKLearnProcessor(
    framework_version="0.20.0",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=role,
)

sklearn_processor.run(
    code="code/preprocessing.py",
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data, destination="/opt/ml/processing/input/"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            source="/opt/ml/processing/train/", output_name="train_data"
        ),
        sagemaker.processing.ProcessingOutput(
            source="/opt/ml/processing/test/", output_name="test_data"
        ),
    ],
    arguments=["--train-test-split-ratio", "0.2"],
)
```

`code/preprocessing.py`
1. load data
2. do stuff
3. save data

## Training jobs
```python
import sagemaker.sklearn.estimator

sklearn = sagemaker.sklearn.estimator.SKLearn(
    entry_point="code/train.py",
    framework_version="0.20.0",
    instance_type="ml.m5.xlarge",
    role=role,
)
sklearn.fit({"train": preprocessed_training_data},)
```

`code/train.py` steps:
1. load data
2. train model
3. save model

### Distributed training
XGBoost images can train on multiple instances but create a single model artifact. You shard the s3 data between instances, each runs the same train script but XGBoost has knowledge of the other workers and combines to a single model. With custom images we can install XGBoost but additional config is required to setup distributed training. Distributed training is not supported with Sklearn, but is available with LightGBM and Pytorch/Lightning.

## Pipelines

## Projects

## Autogluon

## Model monitor


## Data monitoring
Clarify

## Studio
1. What does it offer not available through SDK?