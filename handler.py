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

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = os.environ.get('AWS_BUCKET_NAME', 'chuck-assets')

# Get model from the mounted S3 bucket in the Runpod Volume
MODEL_PATH = "/runpod-volume"

# Initialize your model globally (only loaded once per container)
device = "cuda" if torch.cuda.is_available() else "cpu"
shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    MODEL_PATH,
    subfolder="",
    use_safetensors=True,
    device=device,
)

def handler(event):
    print("Worker Start")
    input_data = event['input']

    # Get base64 encoded image from input
    image_b64 = input_data.get('image_base64')

    # Get user ID from input
    user_id = input_data.get('user_id')

    # If no user ID is provided, stop processing
    if not user_id:
        return {'status': 'error', 'message': 'No user ID provided'}

    if not image_b64:
        return {'status': 'error', 'message': 'No image provided'}

    # Generate one single UUID for both files
    file_id = str(uuid.uuid4())

    # Decode the base64 image
    image_bytes = base64.b64decode(image_b64)

    # Load image into PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Save input image to temp file for S3 upload
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_image:
        image.save(temp_image.name)
        temp_image.flush()

        # Upload input image to S3
        image_key = f"images/{file_id}.png"
        s3.upload_file(temp_image.name, bucket_name, image_key)

    # Run inference
    with torch.inference_mode():
        result = shape_pipe(
            image=image,
            num_inference_steps=10,
            octree_resolution=180,
            num_chunks=60000,
            generator=torch.manual_seed(12355),
            output_type="trimesh"
        )

    # Save STL to temp file
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_stl:
        result[0].export(temp_stl.name, file_type="stl")
        stl_path = temp_stl.name

    # Upload STL to S3
    stl_key = f"models/{file_id}.stl"
    s3.upload_file(stl_path, bucket_name, stl_key)

    # Generate presigned URLs
    image_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': image_key},
        ExpiresIn=3600
    )
    stl_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': stl_key},
        ExpiresIn=3600
    )

    return {
        'status': 'success',
        'uuid': file_id,
        'image_url': image_url,
        'stl_url': stl_url,
        'user_id': user_id
    }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
