import os
from wan2gp_api import Wan2GPAPI

def test_api():
    # Initialize API with a specific output directory
    api = Wan2GPAPI(output_dir="test_outputs")
    
    # Text for testing
    prompt = "A majestic dragon flying over a crystalline lake, high quality, 4k"
    
    print("\n--- Testing Flux 2 Klein 9B ---")
    try:
        # Note: In a real environment, this would actually run the models.
        # We are using a low step count for faster (hypothetical) testing.
        flux_image = api.generate_flux(
            prompt=prompt,
            steps=4,
            resolution="512x512" # Smaller resolution for testing
        )
        print(f"Flux result: {flux_image}")
    except Exception as e:
        print(f"Flux generation failed: {e}")

    print("\n--- Testing LTX 2.3 Distilled ---")
    try:
        # Testing video generation with minimal frames
        video_result = api.generate_video(
            prompt="Simple motion of water ripples",
            steps=10,
            video_length=17, # Minimum frames for LTX
            resolution="480x272"
        )
        print(f"LTX result: {video_result}")
    except Exception as e:
        print(f"LTX generation failed: {e}")

if __name__ == "__main__":
    test_api()
