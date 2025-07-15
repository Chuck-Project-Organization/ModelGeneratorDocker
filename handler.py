import runpod
import boto3
import tempfile
import base64
import os
import torch
from PIL import Image
import io
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import requests
from datetime import datetime, timezone
import pymeshlab


def log(message, level="ℹ️"):
    print(f"{level} [{datetime.now(timezone.utc).isoformat()}] {message}")

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = os.environ.get('AWS_BUCKET_NAME', 'chuck-assets')
webhook_secret = os.environ.get('WEBHOOK_SECRET')
webhook_url = os.environ.get('WEBHOOK_URL')

log(f"Loaded environment: bucket={bucket_name}, webhook_url={webhook_url}")

# Model path and device setup
MODEL_PATH = "/runpod-volume"
device = "cuda" if torch.cuda.is_available() else "cpu"

log(f"Loading model from {MODEL_PATH} on {device}")
shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    MODEL_PATH,
    subfolder="",
    use_safetensors=True,
    device=device,
)

def handler(job):
    try:
        log("🟢 Worker started")

        file_id = job.get('id')
        log(f"Files ID determined as job ID: {file_id}")

        input_data = job.get('input', {})
        user_id = input_data.get('user_id')
        image_b64 = input_data.get('image_base64')

        if not user_id:
            log("Missing user_id", "❌")
            return {'status': 'error', 'message': 'No user ID provided'}

        if not image_b64:
            log("Missing image", "❌")
            return {'status': 'error', 'message': 'No image provided'}

        # Decode base64 and prepare image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".png") as temp_image:
            image.save(temp_image.name)
            image_key = f"images/{file_id}.png"
            s3.upload_file(temp_image.name, bucket_name, image_key)
            log(f"📤 Uploaded input image to s3://{bucket_name}/{image_key}")

        with torch.inference_mode():
            log("🧠 Running inference...")
            result = shape_pipe(
                image=image,
                num_inference_steps=10,
                # Boosting the octree resolution both increases file size and time to generate
                # by a lot. We should avoid it
                octree_resolution=128,
                num_chunks=60000,
                generator=torch.manual_seed(12355),
                output_type="trimesh"
            )

        # Post-process the mesh
        log("🔧 Post-processing mesh...")
        if not result or len(result) == 0:
            raise ValueError("No mesh generated from the input image.")

        # Save original mesh to OBJ for pymeshlab
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as temp_obj:
            result[0].export(temp_obj.name, file_type="obj")
            obj_path = temp_obj.name

        # Use PyMeshLab to smooth/refine
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_path)

        # Apply Catmull-Clark subdivision and smoothing
        ms.apply_filter('subdivision_surfaces_catmull_clark', iterations=1)
        ms.apply_filter('laplacian_smooth', iteration=5)

        # Save to STL for S3
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_stl:
            ms.save_current_mesh(temp_stl.name, file_format='stl')
            stl_path = temp_stl.name

        stl_key = f"models/{file_id}.stl"
        s3.upload_file(stl_path, bucket_name, stl_key)
        log(f"📤 Uploaded STL model to s3://{bucket_name}/{stl_key}")

        image_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': image_key}, ExpiresIn=3600)
        stl_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': stl_key}, ExpiresIn=3600)

        body = {
            'status': 'success',
            'job_id': file_id,
            'image_url': image_url,
            'stl_url': stl_url,
            'user_id': user_id
        }

        headers = {
            'X-Auth-Token': webhook_secret
        }

        log(f"📡 Sending webhook to {webhook_url}")
        response = requests.post(
            webhook_url,
            json=body,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        log(f"✅ Webhook success — status {response.status_code}")
        log(f"➡️ Webhook payload: {body}")

        return body

    except Exception as e:
        log(f"❌ Fatal error: {str(e)}", "❌")
        raise

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})