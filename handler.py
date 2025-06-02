import runpod
import boto3
import tempfile
import base64
import os
import uuid

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = os.environ.get('AWS_BUCKET_NAME', 'chuck-assets')

def handler(event):
    '''
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        dict: The result to be returned to the client
    '''
    print(f"Worker Start")
    input_data = event['input']

    # Get base64 encoded image from input
    image_b64 = input_data.get('image_base64')
    if not image_b64:
        return {'status': 'error', 'message': 'No image provided'}

    # Decode the base64 image
    image_bytes = base64.b64decode(image_b64)

    # Write image to temporary file
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_image:
        temp_image.write(image_bytes)
        temp_image.flush()

        # Create unique key for S3
        key = f"images/{uuid.uuid4()}.png"

        # Upload to S3
        print(f"Uploading {temp_image.name} to S3 bucket {bucket_name}...")
        s3.upload_file(temp_image.name, bucket_name, key)

        # Generate presigned URL valid for 1 hour
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': key},
            ExpiresIn=3600
        )

    return {
        'status': 'success',
        's3_url': presigned_url
    }

# import base64
# from PIL import Image
# import io
# import torch
# from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# shape_pipe = None

# def handler(event):
#     global shape_pipe

#     try:
#         if shape_pipe is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
#                 "tencent/Hunyuan3D-2mini",
#                 subfolder="hunyuan3d-dit-v2-mini-turbo",
#                 use_safetensors=True,
#                 device=device,
#             )

#         input_data = event["input"]
#         image_b64 = input_data["image"]
#         filename = input_data.get("filename", "model.stl")

#         # Decode image
#         image_bytes = base64.b64decode(image_b64)
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#         # Generate model
#         with torch.inference_mode():
#             result = shape_pipe(
#                 image=image,
#                 num_inference_steps=10,
#                 octree_resolution=180,
#                 num_chunks=60000,
#                 generator=torch.manual_seed(12355),
#                 output_type="trimesh"
#             )

#         # Export to STL and encode
#         buffer = io.BytesIO()
#         result[0].export(buffer, file_type="stl")
#         buffer.seek(0)
#         stl_b64 = base64.b64encode(buffer.read()).decode("utf-8")

#         return {
#             "output": {
#                 "filename": filename,
#                 "stl": stl_b64
#             }
#         }

#     except Exception as e:
#         return {
#             "error": str(e)
#         }
