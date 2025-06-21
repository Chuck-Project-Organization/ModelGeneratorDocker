import runpod
import boto3
import tempfile
import base64
import os
import uuid
import torch
from PIL import Image
import io
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import requests
from datetime import datetime

def log(message, level="ℹ️"):
    print(f"{level} [{datetime.utcnow().isoformat()}] {message}")

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

def handler(event):
    try:
        log("🟢 Worker started")

        input_data = event.get('input', {})
        user_id = input_data.get('user_id')
        image_b64 = input_data.get('image_base64')

        if not user_id:
            log("Missing user_id", "❌")
            return {'status': 'error', 'message': 'No user ID provided'}

        if not image_b64:
            log("Missing image", "❌")
            return {'status': 'error', 'message': 'No image provided'}

        file_id = str(uuid.uuid4())
        log(f"Generated file_id: {file_id}")

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
                octree_resolution=180,
                num_chunks=60000,
                generator=torch.manual_seed(12355),
                output_type="trimesh"
            )

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_stl:
            result[0].export(temp_stl.name, file_type="stl")
            stl_path = temp_stl.name

        stl_key = f"models/{file_id}.stl"
        s3.upload_file(stl_path, bucket_name, stl_key)
        log(f"📤 Uploaded STL model to s3://{bucket_name}/{stl_key}")

        image_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': image_key}, ExpiresIn=3600)
        stl_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': stl_key}, ExpiresIn=3600)

        body = {
            'status': 'success',
            'uuid': file_id,
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