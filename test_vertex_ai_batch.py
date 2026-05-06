import os
import time
from datetime import datetime

import fsspec
import pandas as pd
from google import genai
from google.genai import types
from google.genai.types import CreateBatchJobConfig

client = genai.Client(
    vertexai=True,
    project='project-417aacc6-490b-43c9-9da',
    location='global',
    http_options=types.HttpOptions(api_version='v1')
)

INPUT_DATA = "gs://bucket-lic/input_file_batch/batch_requests_for_multimodal_input_2.jsonl" #"gs://cloud-samples-data/generative-ai/batch/batch_requests_for_multimodal_input_2.jsonl"  # @param {type:"string"}

BUCKET_URI = "gs://bucket-lic"  # @param {type:"string"}

#     ! gcloud storage buckets create {BUCKET_URI} --project={PROJECT_ID} --location={GCS_LOCATION}
MODEL_ID = "gemini-3-flash-preview"  # @param {type:"string"}
gcs_batch_job = client.batches.create(
    model=MODEL_ID,
    src=INPUT_DATA,
    config=CreateBatchJobConfig(dest=BUCKET_URI),
)
print(f"Batch job created: {gcs_batch_job.name}")
gcs_batch_job = client.batches.get(name=gcs_batch_job.name)

# Refresh the job until complete
while gcs_batch_job.state in (
    "JOB_STATE_RUNNING",
    "JOB_STATE_PENDING",
    "JOB_STATE_QUEUED",
):
    print(f"Job state: {gcs_batch_job.state}. Waiting for completion...")
    time.sleep(30)
    gcs_batch_job = client.batches.get(name=gcs_batch_job.name)

# Check if the job succeeds
if gcs_batch_job.state == "JOB_STATE_SUCCEEDED":
    print("Job succeeded!")
    print(f"Output predictions are located at: {gcs_batch_job.dest.gcs_uri}/*/predictions.jsonl")
else:
    print(f"Job failed: {gcs_batch_job.error}")



# The example code below shows how to load the `.jsonl` file in the Cloud Storage output location into a Pandas DataFrame and print out the object.

# You can retrieve the specific responses in the `response` field.
fs = fsspec.filesystem("gcs")

file_paths = fs.glob(f"{gcs_batch_job.dest.gcs_uri}/*/predictions.jsonl")

if gcs_batch_job.state == "JOB_STATE_SUCCEEDED":
    # Load the JSONL file into a DataFrame
    df = pd.read_json(f"gs://{file_paths[0]}", lines=True)

    df = df.join(pd.json_normalize(df["response"], "candidates"))
    display(df)