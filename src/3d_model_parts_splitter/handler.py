import runpod
import boto3
import tempfile
import os
import torch
import requests
import subprocess
import shutil
import uuid
import base64
from urllib.request import urlretrieve
from datetime import datetime, timezone

# ---------- logging ----------
def log(message, level="ℹ️"):
    print(f"{level} [{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)

# ---------- env & clients ----------
s3 = boto3.client('s3')
bucket_name = os.environ.get('AWS_BUCKET_NAME', 'chuck-assets')
webhook_secret = os.environ.get('WEBHOOK_SECRET')
webhook_url = os.environ.get('WEBHOOK_URL')

PF_ROOT = os.getenv("PF_ROOT", "/workspace/PartField/partfield")  # folder with scripts
PF_CKPT = os.getenv("PF_CKPT", "/runpod-volume/model/model_objaverse.ckpt")  # read-only on volume
device = "cuda" if torch.cuda.is_available() else "cpu"

log(f"Loaded environment: bucket={bucket_name}, webhook_url={webhook_url}, PF_ROOT={PF_ROOT}, PF_CKPT={PF_CKPT}, device={device}")

# ---------- helpers ----------
def run_cmd(cmd, cwd=None):
    """Run shell command and stream logs; raise on failure."""
    log(f"▶️ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)

def ensure_paths():
    if not os.path.isdir(PF_ROOT):
        raise RuntimeError(f"PF_ROOT does not exist: {PF_ROOT}")
    if not os.path.isfile(PF_CKPT):
        raise RuntimeError(f"PF_CKPT not found at: {PF_CKPT}")

def download_to(path, url=None, content_bytes=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if url:
        urlretrieve(url, path)
    elif content_bytes:
        with open(path, "wb") as f:
            f.write(content_bytes)
    else:
        raise ValueError("Provide url or content_bytes")
    return path

def zip_dir(path):
    zip_path = f"{path}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(path, "zip", path)
    return zip_path

def s3_upload(local_path, key, expires=3600):
    s3.upload_file(local_path, bucket_name, key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": key},
        ExpiresIn=expires
    )
    log(f"📤 Uploaded to s3://{bucket_name}/{key}")
    return url

# ---------- PartField wrappers ----------
def partfield_inference(job_id, data_dir, preprocess=True):
    """
    Runs feature extraction; outputs under PF_ROOT/exp_results/partfield_features/<job_id>
    """
    result_name = f"partfield_features/{job_id}"
    cmd = [
        "python", "partfield_inference.py",
        "-c", "configs/final/demo.yaml",
        "--opts",
        "continue_ckpt", PF_CKPT,          # read-only checkpoint on volume
        "result_name", result_name,
        "dataset.data_path", data_dir,     # absolute /tmp path with the STL
    ]
    if preprocess:
        cmd += ["preprocess_mesh", "True"]
    run_cmd(cmd, cwd=PF_ROOT)
    return os.path.join(PF_ROOT, "exp_results", "partfield_features", job_id)

def partfield_clustering(job_id, data_dir, feat_root, mode="agglo_knn", max_clusters=20):
    """
    Clusters features into parts; outputs under PF_ROOT/exp_results/clustering/<job_id>
    mode: "agglo" | "agglo_knn" | "kmeans"
    """
    dump_dir = os.path.join(PF_ROOT, "exp_results", "clustering", job_id)
    os.makedirs(dump_dir, exist_ok=True)

    base = [
        "python", "run_part_clustering.py",
        "--root", feat_root,
        "--dump_dir", dump_dir,
        "--source_dir", data_dir,
        "--max_num_clusters", str(max_clusters),
    ]
    if mode == "agglo":
        cmd = base + ["--use_agglo", "True", "--option", "0"]
    elif mode == "agglo_knn":
        cmd = base + ["--use_agglo", "True", "--option", "1", "--with_knn", "True"]
    elif mode == "kmeans":
        cmd = base  # default codepath = kmeans (no adjacency)
    else:
        raise ValueError("mode must be one of: agglo, agglo_knn, kmeans")

    run_cmd(cmd, cwd=PF_ROOT)
    return dump_dir

# ---------- handler ----------
def handler(job):
    job_dir = None
    try:
        log("🟢 Worker started")
        ensure_paths()

        file_id = job.get('id') or str(uuid.uuid4())[:8]
        log(f"Files ID determined as job ID: {file_id}")

        # Get input data from job
        input_data = job.get('input', {})
        user_id = input_data.get('user_id')
        stl_presigned_url = input_data.get('stl_presigned_url')
        mesh_url = input_data.get('mesh_url')            # optional alt key
        mesh_b64 = input_data.get('mesh_base64')         # optional alt key
        filename = input_data.get('filename', 'input.stl')

        mode = input_data.get('mode', 'agglo_knn')       # "agglo" | "agglo_knn" | "kmeans"
        max_k = int(input_data.get('max_num_clusters', 20))

        if not user_id:
            log("Missing user_id", "❌")
            return {'status': 'error', 'message': 'No user ID provided'}

        # Create ephemeral scratch space
        job_dir = tempfile.mkdtemp(prefix=f"job_{file_id}_", dir="/tmp")
        data_dir = os.path.join(job_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        local_mesh_path = os.path.join(data_dir, filename)

        # 1) Get STL file (from request)
        if stl_presigned_url:
            log("⬇️ Downloading STL from presigned URL")
            download_to(local_mesh_path, url=stl_presigned_url)
        elif mesh_url:
            log("⬇️ Downloading mesh from mesh_url")
            download_to(local_mesh_path, url=mesh_url)
        elif mesh_b64:
            log("⬇️ Writing mesh from base64")
            download_to(local_mesh_path, content_bytes=base64.b64decode(mesh_b64))
        else:
            return {'status': 'error', 'message': 'No mesh provided (stl_presigned_url | mesh_url | mesh_base64)'}

        log(f"📄 Mesh saved to {local_mesh_path}")

        # 2) Call pipeline inputting STL file
        log("🧠 Step 1/2: Inference (feature extraction)")
        feat_root = partfield_inference(file_id, data_dir, preprocess=True)
        log(f"✅ Features at {feat_root}")

        log("🧩 Step 2/2: Clustering (segmentation)")
        cluster_dir = partfield_clustering(file_id, data_dir, feat_root, mode=mode, max_clusters=max_k)
        log(f"✅ Clustering output at {cluster_dir}")

        # 3) Save (upload) results to S3
        # 3a) Upload original STL
        stl_key = f"splitted_models/{file_id}/{filename}"
        stl_url = s3_upload(local_mesh_path, stl_key)

        # 3b) Zip clustering outputs and upload
        log("🗜️ Zipping clustering results")
        zip_path = zip_dir(cluster_dir)
        zip_key = f"splitted_models/{file_id}/partfield_clustering.zip"
        clustering_zip_url = s3_upload(zip_path, zip_key)

        # 4) Create body response
        body = {
            'status': 'success',
            'job_id': file_id,
            'user_id': user_id,
            'input_mesh_url': stl_url,
            'clustering_zip_url': clustering_zip_url,
            'mode': mode,
            'max_num_clusters': max_k,
        }

        # 5) Log + webhook + return
        if webhook_url:
            headers = {'X-Auth-Token': webhook_secret} if webhook_secret else {}
            log(f"📡 Sending webhook to {webhook_url}")
            response = requests.post(webhook_url, json=body, headers=headers, timeout=15)
            response.raise_for_status()
            log(f"✅ Webhook success — status {response.status_code}")
            log(f"➡️ Webhook payload: {body}")
        else:
            log("ℹ️ WEBHOOK_URL not set; skipping webhook")

        return body

    except Exception as e:
        log(f"❌ Fatal error: {str(e)}", "❌")
        raise
    finally:
        # Clean up ephemeral scratch
        if job_dir and os.path.isdir(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
            log(f"🧹 Cleaned {job_dir}")

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
