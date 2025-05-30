import base64
from PIL import Image
import io
import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mini",
    subfolder="hunyuan3d-dit-v2-mini-turbo",
    use_safetensors=True,
    device=device,
)

def handler(event):
    try:
        input_data = event["input"]
        image_b64 = input_data["image"]
        filename = input_data.get("filename", "model.stl")

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Generate model
        with torch.inference_mode():
            result = shape_pipe(
                image=image,
                num_inference_steps=10,
                octree_resolution=180,
                num_chunks=60000,
                generator=torch.manual_seed(12355),
                output_type="trimesh"
            )

        # Export to STL and encode
        buffer = io.BytesIO()
        result[0].export(buffer, file_type="stl")
        buffer.seek(0)
        stl_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "output": {
                "filename": filename,
                "stl": stl_b64
            }
        }

    except Exception as e:
        return {
            "error": str(e)
        }
