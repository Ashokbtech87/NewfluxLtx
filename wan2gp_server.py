import os
import io
import base64
import traceback
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import uvicorn

# Import the API wrapper we created
from wan2gp_api import Wan2GPAPI

app = FastAPI(
    title="Wan2GP Headless API",
    description="REST API for Wan2GP image and video generation",
    version="1.0.0"
)

# Initialize the API engine
# Using a specific output directory for API generated files
API_OUTPUT_DIR = "api_outputs"
try:
    wan2gp_engine = Wan2GPAPI(output_dir=API_OUTPUT_DIR)
    print("Wan2GPAPI engine initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Wan2GPAPI: {e}")
    traceback.print_exc()
    wan2gp_engine = None

class LoraRequest(BaseModel):
    path: str = Field(description="Name or absolute path to the LoRA safetensors file")
    multiplier: float = Field(default=1.0, description="Strength multiplier for the LoRA")

class FluxRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation, including LoRA trigger words")
    image_start: Optional[str] = Field(default=None, description="Optional path to a starting image")
    image_refs: Optional[List[str]] = Field(default=None, description="Optional list of reference image paths")
    loras: Optional[List[LoraRequest]] = Field(default=None, description="List of LoRAs to apply")
    resolution: str = Field(default="1024x1024", description="Image resolution, e.g., '1024x1024'")
    steps: int = Field(default=20, description="Number of inference steps")
    guidance_scale: float = Field(default=3.5, description="Guidance scale (CFG)")
    seed: int = Field(default=-1, description="Random seed, -1 for random")

class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    image_start: Optional[str] = Field(default=None, description="Path to the starting image frame")
    image_end: Optional[str] = Field(default=None, description="Path to the ending image frame")
    loras: Optional[List[LoraRequest]] = Field(default=None, description="List of LoRAs to apply")
    resolution: str = Field(default="1280x720", description="Video resolution, e.g., '1280x720'")
    steps: int = Field(default=30, description="Number of inference steps")
    guidance_scale: float = Field(default=3.0, description="Guidance scale (CFG)")
    video_length: int = Field(default=81, description="Number of frames (LTX 2.3 default is 97 or 81 depending on mode)")
    seed: int = Field(default=-1, description="Random seed, -1 for random")

def _encode_file_to_base64(filepath: str) -> str:
    """Read a file and return its base64 string representation."""
    if not filepath or not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded

def _format_loras(lora_requests: Optional[List[LoraRequest]]) -> Optional[List[tuple]]:
    if not lora_requests:
        return None
    return [(lora.path, lora.multiplier) for lora in lora_requests]

@app.get("/api/models")
async def list_models():
    """List available architectures known to the system."""
    if not wan2gp_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    return {"models": wan2gp_engine.list_models()}

@app.post("/api/generate/flux")
async def generate_flux(request: FluxRequest):
    """Generate an image using Flux 2 Klein 9B and return the Base64 encoded result."""
    if not wan2gp_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
        
    try:
        loras = _format_loras(request.loras)
        result_path = wan2gp_engine.generate_flux(
            prompt=request.prompt,
            image_start=request.image_start,
            image_refs=request.image_refs,
            loras=loras,
            resolution=request.resolution,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="Generation completed but output file not found.")
            
        base64_data = _encode_file_to_base64(result_path)
        return {
            "status": "success",
            "file_path": result_path, # Server path for reference
            "file_name": os.path.basename(result_path),
            "file_data_base64": base64_data
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/video")
async def generate_video(request: VideoRequest):
    """Generate a video using LTX 2.3 Distilled and return the Base64 encoded result."""
    if not wan2gp_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
        
    try:
        loras = _format_loras(request.loras)
        result_path = wan2gp_engine.generate_video(
            prompt=request.prompt,
            image_start=request.image_start,
            image_end=request.image_end,
            loras=loras,
            resolution=request.resolution,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            video_length=request.video_length,
            seed=request.seed
        )
        
        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="Generation completed but output file not found.")
            
        base64_data = _encode_file_to_base64(result_path)
        return {
            "status": "success",
            "file_path": result_path,
            "file_name": os.path.basename(result_path),
            "file_data_base64": base64_data
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Bind to 0.0.0.0 so it can be exposed via locatunnel or ngrok from Colab
    # Port 8000 is default, feel free to change
    port = int(os.environ.get("SERVER_PORT", 8000))
    print(f"Starting Wan2GP Headless API on 0.0.0.0:{port}")
    uvicorn.run("wan2gp_server:app", host="0.0.0.0", port=port, log_level="info")

