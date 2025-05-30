import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
import os
import io

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
shape_pipe = None

# Load the model just once at startup
@app.on_event("startup")
def load_pipeline():
    global shape_pipe
    print("📦 Loading 3D generation pipeline...")
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        use_safetensors=True,
        device=device,
    )
    print("✅ Pipeline ready.")

@app.post("/generate/")
async def generate_model(file: UploadFile = File(...)):
    os.makedirs("./models", exist_ok=True)
    filename = file.filename
    file_bytes = await file.read()

    # Open uploaded image from memory
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    print(f"🧊 Generating 3D model from {filename}...")
    with torch.inference_mode():
        result = shape_pipe(
            image=image,
            num_inference_steps=10,
            octree_resolution=180,
            num_chunks=60000,
            generator=torch.manual_seed(12355),
            output_type="trimesh"
        )

    mesh = result[0]
    output_name = filename.rsplit(".", 1)[0] + ".stl"
    output_path = f"./models/{output_name}"
    mesh.export(output_path)

    print(f"✅ STL exported: {output_path}")
    return FileResponse(output_path, filename=output_name)
