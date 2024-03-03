# import os
# from typing import Any, Dict, List

# import google.cloud.aiplatform as aip
# import kfp
# from kfp.v2 import compiler  

# project_number = "first-project-413614"

# # Get your Google Cloud project ID from gcloud
# BUCKET_NAME="gs://" + project_number + "-bucket1"

# BUCKET_URI="gs://" + project_number + "-bucket"



# PIPELINE_ROOT = f"{BUCKET_URI}"

# hp_dict: str = '{"num_hidden_layers": 3, "hidden_size": 32, "learning_rate": 0.01, "epochs": 1, "steps_per_epoch": -1}'
# data_dir: str = "gs://first-project-413614-bucket/bikes_weather/flower_photos/"
# TRAINER_ARGS = ["--data-dir", data_dir, "--hptune-dict", hp_dict]

# # create working dir to pass to job spec
# WORKING_DIR = f"{PIPELINE_ROOT}"

# MODEL_DISPLAY_NAME = f"train_deploy"
# print(TRAINER_ARGS, WORKING_DIR, MODEL_DISPLAY_NAME)


# @kfp.dsl.pipeline(name="train-build-deploy")
# def pipeline(
#     project: str = project_number,
#     model_display_name: str = MODEL_DISPLAY_NAME,
#     serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
# ):
#     from google_cloud_pipeline_components.types import artifact_types
#     from google_cloud_pipeline_components.v1.custom_job import \
#         CustomTrainingJobOp
#     from kfp.v2.components import importer_node
#     from kfp import dsl


#     from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp,
#                                                               ModelDeployOp)
#     from google_cloud_pipeline_components.v1.model import ModelUploadOp
#     custom_job_task = CustomTrainingJobOp(
#         project=project,
#         display_name="model-training",
#         worker_pool_specs=[
#             {
#                 "containerSpec": {
#                     "env": [{"name": "flower classification", "value": WORKING_DIR}],
#                     "imageUri": "us-central1-docker.pkg.dev/first-project-413614/flower-app/flower_image:latest",
#                 },
#                 "replicaCount": "1",
#                 "machineSpec": {
#                     "machineType": "n1-standard-8",
#                 },
#             }
#         ],
#     )
    


#     import_unmanaged_model_task = importer_node.importer(
#         artifact_uri=f"{BUCKET_URI}/model",
#         artifact_class=artifact_types.UnmanagedContainerModel,
#         metadata={
#             "containerSpec": {
#                 "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
#             },
#         },
#     ).after(custom_job_task)

#     model_upload_op = ModelUploadOp(
#         project=project,
#         display_name=model_display_name,
#         unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
#     )
#     model_upload_op.after(import_unmanaged_model_task)

#     endpoint_create_op = EndpointCreateOp(
#         project=project,
#         display_name="pipelines-created-endpoint",
#     )

#     ModelDeployOp(
#         endpoint=endpoint_create_op.outputs["endpoint"],
#         model=model_upload_op.outputs["model"],
#         deployed_model_display_name=model_display_name,
#         dedicated_resources_machine_type="n1-standard-4",
#         dedicated_resources_min_replica_count=1,
#         dedicated_resources_max_replica_count=1,
#     )

# compiler.Compiler().compile(
#     pipeline_func=pipeline,
#     package_path="tabular_regression_pipeline.json",
# )
# DISPLAY_NAME = "bikes_weather_"

# job = aip.PipelineJob(
#     display_name=DISPLAY_NAME,
#     template_path="tabular_regression_pipeline.json",
#     pipeline_root=PIPELINE_ROOT,
#     enable_caching=False,
# )

# job.run()

# import os
# from typing import Any, Dict, List

# import kfp
# import google.cloud.aiplatform as aip

# from kfp.v2 import compiler  

# project_number = "first-project-413614"

# # Get your Google Cloud project ID from gcloud
# BUCKET_NAME="gs://" + project_number + "-bucket1"

# BUCKET_URI="gs://" + project_number + "-bucket"

# PIPELINE_ROOT = f"{BUCKET_URI}"

# hp_dict: str = '{"num_hidden_layers": 3, "hidden_size": 32, "learning_rate": 0.01, "epochs": 1, "steps_per_epoch": -1}'
# data_dir: str = "gs://first-project-413614-bucket/bikes_weather/flower_photos/"
# TRAINER_ARGS = ["--data-dir", data_dir, "--hptune-dict", hp_dict]

# # create working dir to pass to job spec
# WORKING_DIR = f"{PIPELINE_ROOT}"

# MODEL_DISPLAY_NAME = f"train_deploy"
# print(TRAINER_ARGS, WORKING_DIR, MODEL_DISPLAY_NAME)


# @kfp.dsl.pipeline(name="train-model")
# def pipeline(
#     project: str = project_number,
# ):
#     from google_cloud_pipeline_components.types import artifact_types
#     from google_cloud_pipeline_components.v1.custom_job import \
#         CustomTrainingJobOp
#     from kfp import dsl


#     custom_job_task = CustomTrainingJobOp(
#         project=project,
#         display_name="model-training",
#         worker_pool_specs=[
#             {
#                 "containerSpec": {
#                     "env": [{"name": "flower classification", "value": WORKING_DIR}],
#                     "imageUri": "us-central1-docker.pkg.dev/first-project-413614/flower-app/flower_image:latest",
#                 },
#                 "replicaCount": "1",
#                 "machineSpec": {
#                     "machineType": "n1-standard-8",
#                 },
#             }
#         ],
#     )
    
# compiler.Compiler().compile(
#     pipeline_func=pipeline,
#     package_path="train_model_pipeline.json",
# )
# DISPLAY_NAME = "bikes_weather_"

# job = aip.PipelineJob(
#     display_name=DISPLAY_NAME,
#     template_path="train_model_pipeline.json",
#     pipeline_root=PIPELINE_ROOT,
#     enable_caching=False,
# )

# job.run()
import kfp
from kfp.v2 import compiler
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1 import hyperparameter_tuning_job
project_number = "first-project-413614"

BUCKET_NAME="gs://" + project_number + "-bucket1"

BUCKET_URI="gs://" + project_number + "-bucket"

PIPELINE_ROOT = f"{BUCKET_URI}"


# Define hyperparameters and data directory
hp_dict = '{"num_hidden_layers": 3, "hidden_size": 32, "learning_rate": 0.01, "epochs": 1, "steps_per_epoch": -1}'
data_dir = f"gs://{project_number}-bucket/bikes_weather/flower_photos/"

# Define the hyperparameter tuning job
def tuning_job(project: str, display_name: str, hp_dict: str, data_dir: str):
    return hyperparameter_tuning_job.HyperparameterTuningJobRunOp(
        project=project,
        display_name=display_name,
        container_uri="us-central1-docker.pkg.dev/first-project-413614/flower-app/flower_image:latest",
        args=["--data-dir", data_dir, "--hptune-dict", hp_dict],
        metric="accuracy",
        max_trial_count=10,
        parallel_trial_count=3,
        machine_spec={"machine_type": "n1-standard-8"},
        objective="maximize",
        network={"use_current_subnet": True},
    )

# Define the training job with hyperparameters as arguments
def training_job(project: str, display_name: str, data_dir: str, num_hidden_layers: int, hidden_size: int, learning_rate: float, epochs: int, steps_per_epoch: int):
    return CustomTrainingJobOp(
        project=project,
        display_name=display_name,
        worker_pool_specs=[
            {
                "containerSpec": {
                    "env": [
                        {"name": "flower classification", "value": data_dir},
                        {"name": "num_hidden_layers", "value": str(num_hidden_layers)},
                        {"name": "hidden_size", "value": str(hidden_size)},
                        {"name": "learning_rate", "value": str(learning_rate)},
                        {"name": "epochs", "value": str(epochs)},
                        {"name": "steps_per_epoch", "value": str(steps_per_epoch)}
                    ],
                    "imageUri": "us-central1-docker.pkg.dev/first-project-413614/flower-app/flower_image:latest",
                },
                "replicaCount": "1",
                "machineSpec": {
                    "machineType": "n1-standard-8",
                },
            }
        ],
    )

# Define the pipeline
@kfp.dsl.pipeline(name="train-model")
def pipeline(
    project: str = project_number,
):

    # Hyperparameter tuning job
    tuning_task = tuning_job(
        project=project,
        display_name="hyperparameter-tuning",
        hp_dict=hp_dict,
        data_dir=data_dir
    )

    # Training job
    training_task = training_job(
        project=project,
        display_name="model-training",
        data_dir=data_dir,
        num_hidden_layers=tuning_task.outputs["best_hyperparameters"]["num_hidden_layers"],
        hidden_size=tuning_task.outputs["best_hyperparameters"]["hidden_size"],
        learning_rate=tuning_task.outputs["best_hyperparameters"]["learning_rate"],
        epochs=tuning_task.outputs["best_hyperparameters"]["epochs"],
        steps_per_epoch=tuning_task.outputs["best_hyperparameters"]["steps_per_epoch"]
    )

    # The training job should start only after the tuning job completes
    training_task.after(tuning_task)

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="train_model_pipeline.json",
)

# Run the pipeline
job = kfp.v2.client.Client().create_run_from_job_spec(
    job_spec_path="train_model_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
)
