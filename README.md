# Overview

Repository to hold code that will be built in Runpod. Each module under src is used to create a different Serverless Endpoint.

## How to Use Volumes

Serverless Endpoints leverage volumes by fetching large files from it in a low-latency transaction. Every time a Serverless instance starts up, it goes through a cold start, meaning it has to download all the dependencies before being ready to be used. Since we are dealing with ML Models (large files, GB in size), having to download them every time an instance cold start, it would take minutes before the job starts being processed (and we pay for that). For this reason, we mount volumes inside the serverless instance, which is much faster and cheaper.

* **How to upload files to Volumes**

Runpod Volumes are AWS S3 based. For this reason, we can use AWS CLI. In the Runpod account, under [Settings](https://console.runpod.io/user/settings) there's is an S3 API Key to access our volumes. We need to use both the Access Key, Secret and the Runpod Volume region and configure them locally using:

```bash
aws configure
```

After that, get you bucket id (same as Volume ID) and region (same as Data Center) in the [Storage](https://console.runpod.io/user/storage) page in the Runpod account. Run the following to upload files to Runpod Volume:

```bash
aws s3 cp <file_name> s3://<s3_bucket_id>/ \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io \
  --cli-connect-timeout 60 \
  --cli-read-timeout 0
```

To check if the file was uploaded successfully, use the following command:

```bash
aws s3 ls s3://<s3_bucket_id>/ \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io --human-readable
```

## How to Develop Serverless Endpoints

Run Docker on your machine and force --platform=linux/amd64 on the build. If we don't pass this platform tag in the command, and we try to build the docker in a MacOS computer for example, Docker will default to another processor architecture and may fail the build.

```bash
docker buildx build --platform=linux/amd64 -t partfield-worker:dev <path-to-folder-with-dockerfile>
```
