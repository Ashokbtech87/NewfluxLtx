# Wan2GP Simplified Python API

The `Wan2GPAPI` provides a headless way to interact with Flux and LTX models for automated story-to-video pipelines.

## Getting Started

```python
from wan2gp_api import Wan2GPAPI

# Initialize
api = Wan2GPAPI(output_dir="outputs")
```

## Image Generation (Flux 2 Klein 9B)

```python
image_path = api.generate_flux(
    prompt="A beautiful sunset over a lake",
    steps=20,
    resolution="1024x1024"
)
```

### Parameters:
- `prompt`: Text description.
- `image_start`: Optional starting image.
- `image_refs`: Optional list of reference images.
- `loras`: Optional list of `(name, weight)` tuples.
- `resolution`: Output size (e.g., "1024x1024").
- `steps`: Inference steps.
- `guidance_scale`: Classifier-free guidance.
- `seed`: Random seed (-1 for random).

## Video Generation (LTX 2.3 Distilled)

```python
video_path = api.generate_video(
    prompt="Water ripples moving outward",
    image_start=image_path,
    video_length=81,
    steps=30
)
```

### Parameters:
- `prompt`: Text description.
- `image_start`: Optional path to start frame.
- `image_end`: Optional path to end frame.
- `loras`: Optional list of `(name, weight)` tuples.
- `resolution`: Output size.
- `steps`: Inference steps.
- `guidance_scale`: Guidance scale.
- `video_length`: Total frames (e.g., 81 for ~3.4s at 24fps).
- `seed`: Random seed.

## Remote execution from Colab to Local PC
If Wan2GP is running on Google Colab and you want to trigger image/video generation from your local PC using Python scripts, you can use the built-in FastAPI server.

**1. Start the Server (in Colab):**
```bash
python wan2gp_server.py
```
*Note: To expose the server to the public internet so your PC can reach it, use `localtunnel`:*
```bash
!npm install -g localtunnel
!npx localtunnel --port 8000
```
This will give you a public URL (e.g. `https://my-colab-url.loca.lt`).

**2. Call the API from your Local PC:**
See `client_example.py` for a full working script. The API returns the generated images and videos encoded as **Base64** strings, so your local script can download and save them directly without any complex file server setups!

## Examples
- **Local Python Script**: See `client_example.py` to learn how to hit the remote API and decode Base64 video responses.
- **Direct Python Import**: See `test_wan2gp_api.py` for a complete working example of a story-to-video pipeline executing directly.
