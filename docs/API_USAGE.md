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

## Examples
See **[test_wan2gp_api.py](../test_wan2gp_api.py)** for a complete working example of a story-to-video pipeline.
