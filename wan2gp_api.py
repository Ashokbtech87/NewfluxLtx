import os
import sys
import torch
import time
import json
import threading
import traceback
from PIL import Image

# Add current directory to path to ensure imports work
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)

import wgp
from shared.utils.utils import has_image_file_extension, has_video_file_extension
from shared.utils.loras_mutipliers import parse_loras_multipliers

class Wan2GPAPI:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Wan2GP environment
        self._setup_environment()
        self.state = self._create_initial_state()
        self.lock = threading.Lock()
        
    def _setup_environment(self):
        # Mirroring wgp.py setup
        wgp.server_config["save_path"] = self.output_dir
        wgp.server_config["image_save_path"] = self.output_dir
        wgp.server_config["audio_save_path"] = self.output_dir
        wgp.server_config["notification_sound_enabled"] = 0
        
        # Ensure ffmpeg is available
        from shared.ffmpeg_setup import download_ffmpeg
        download_ffmpeg()

    def _create_initial_state(self):
        # Minimal state based on wgp.py CLI mode
        return {
            "gen": {
                "queue": [],
                "in_progress": False,
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": 0,
                "audio_selected": 0,
                "prompt_no": 0,
                "prompts_max": 0,
                "repeat_no": 0,
                "total_generation": 1,
                "window_no": 0,
                "total_windows": 0,
                "progress_status": "",
                "process_status": "process:main",
                "abort": False,
            },
            "loras": [],
            "all_settings": {},
            "active_form": "add",
        }

    def list_models(self):
        """List all available model types."""
        return list(wgp._TRANSFORMER_MODELS_DEFS.keys())

    def get_lora_directory(self, model_type):
        """Get the LoRA directory for a given model type."""
        return wgp.get_lora_dir(model_type)

    def generate_flux(self, prompt, image_start=None, image_refs=None, loras=None, resolution="1024x1024", steps=20, guidance_scale=3.5, seed=-1):
        """
        Generate an image using Flux 2 Klein 9B.
        
        loras: list of (lora_name_or_path, multiplier)
        image_refs: list of image paths for multi-image flux
        """
        model_type = "flux2_klein_9b"
        return self._generate(
            model_type=model_type,
            prompt=prompt,
            image_start=image_start,
            image_refs=image_refs,
            loras=loras,
            resolution=resolution,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            image_mode=1 # 1 for Image output
        )

    def generate_video(self, prompt, image_start=None, image_end=None, loras=None, resolution="1280x720", steps=30, guidance_scale=3.0, video_length=81, seed=-1):
        """
        Generate a video using LTX 2.3 Distilled.
        
        loras: list of (lora_name_or_path, multiplier)
        image_start: path to starting frame
        image_end: path to ending frame
        """
        model_type = "ltx2_22B"
        return self._generate(
            model_type=model_type,
            prompt=prompt,
            image_start=image_start,
            image_end=image_end,
            loras=loras,
            resolution=resolution,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            video_length=video_length,
            seed=seed,
            image_mode=0, # 0 for Video output
            ltx2_pipeline="distilled"
        )

    def _generate(self, **kwargs):
        model_type = kwargs.get("model_type")
        params = wgp.get_default_settings(model_type).copy()
        params.update(kwargs)
        params["state"] = self.state
        
        # Handle Seed
        if params.get("seed", -1) <= -1:
            params["seed"] = wgp.random.randint(0, 2**32 - 1)
            
        # Handle LoRAs
        loras = params.get("loras", [])
        if loras:
            activated_loras = []
            loras_multipliers = []
            for lora_path, multiplier in loras:
                activated_loras.append(lora_path)
                loras_multipliers.append(str(multiplier))
            params["activated_loras"] = activated_loras
            params["loras_multipliers"] = "|".join(loras_multipliers)
        else:
            params["activated_loras"] = []
            params["loras_multipliers"] = ""

        # Validate settings using the internal wgp logic
        # This resolves paths, handled LoRAs, and sets up the internal parameters
        validated_params, prompts, image_start, image_end, error = wgp.validate_settings(
            self.state, model_type, True, params, silent=True
        )
        
        if error:
            raise Exception(f"Validation Error: {error}")

        # Add to queue
        with self.lock:
            # We clear the queue first to make it "simple" (sync execution)
            self.state["gen"]["queue"] = []
            wgp.add_video_task(**validated_params)
            if not self.state["gen"]["queue"]:
                raise Exception("Failed to add task to queue")
            task = self.state["gen"]["queue"].pop()
        
        # Process task directly (headless)
        return self._process_task_sync(task)

    def _process_task_sync(self, task):
        gen = self.state["gen"]
        params = task['params']
        
        # Cleanup params for generate_video
        for key in ["model_filename", "lset_name"]:
            params.pop(key, None)
            
        import inspect
        sig = inspect.signature(wgp.generate_video)
        expected_args = set(sig.parameters.keys())
        filtered_params = {k: v for k, v in params.items() if k in expected_args}
        plugin_data = task.get('plugin_data', {})

        # Define a simple command sender for console output
        def send_cmd(cmd_type, data):
            if cmd_type == "status":
                print(f"[Wan2GP] {data}")
            elif cmd_type == "error":
                print(f"[ERROR] {data}")
                raise Exception(data)
            elif cmd_type == "info":
                print(f"[INFO] {data}")

        # Set up GPU resources
        wgp.set_main_generation_running(self.state, True)
        wgp.acquire_main_GPU_ressources(self.state)
        
        try:
            print(f"Starting generation for model: {filtered_params.get('model_type')}")
            # generate_video is a generator in wgp.py
            for progress_info in wgp.generate_video(task, send_cmd, plugin_data=plugin_data, **filtered_params):
                pass
            
            print("Generation process finished.")
            # Return the last generated file if available
            if gen["file_list"]:
                return gen["file_list"][-1]
            return None
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            wgp.set_main_generation_running(self.state, False)
            wgp.release_GPU_ressources(self.state)

if __name__ == "__main__":
    # Example usage
    api = Wan2GPAPI(output_dir="api_outputs")
    print("Available Models:", api.list_models())
    # result = api.generate_flux("A robotic bird in a digital forest", steps=10)
    # print(f"Generated: {result}")
