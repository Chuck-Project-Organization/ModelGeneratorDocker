import runpod
import boto3
import time
import tempfile
import os

s3 = boto3.client('s3')
bucket_name = os.environ.get('AWS_BUCKET_NAME', 'chuck-assets')

def handler(event):
#   This function processes incoming requests to your Serverless endpoint.
#
#    Args:
#        event (dict): Contains the input data and request metadata
#       
#    Returns:
#       Any: The result to be returned to the client
    
    # Extract input data
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    
    # You can replace this sleep call with your own Python code
    time.sleep(seconds)

    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_text:
        temp_text.write(prompt)
        temp_text.flush()

        # Upload the file to S3
        print(f"Uploading {temp_text.name} to S3 bucket {bucket_name}...")
        key = f"prompts/{temp_text.name.split('/')[-1]}"
        s3.upload_file(temp_text.name, bucket_name, key)

        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': key},
            ExpiresIn=3600  # URL valid for 1 hour
        )

    return {
        'status': 'success',
        'message': f'Prompt processed and uploaded to S3: {presigned_url}'
    }

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })

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
