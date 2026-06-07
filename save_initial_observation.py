# ==============================================================================
# PLACEMENT INSTRUCTIONS:
# Place this script in the ROOT directory of your project.
# (This is the same directory where you placed test_env.py, 
# containing the 'capx' and 'env_configs' folders).
#
# RUN INSTRUCTIONS:
# Execute the script from the terminal, passing the path to your YAML config:
# python save_images.py env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml
# ==============================================================================

import os
import sys
import yaml
import numpy as np
from PIL import Image

# Import simulators to ensure all low-level environments are registered
import capx.envs.simulators  
from capx.envs.base import get_env

def main(yaml_path: str, output_dir: str = "initial_observation_images"):
    print(f"Loading config from: {yaml_path}")
    
    # Parse the YAML file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        
    try:
        env_name = config['env']['cfg']['low_level']
    except KeyError as e:
        print(f"Error parsing YAML: Could not find key {e}")
        sys.exit(1)
        
    print(f"Initializing environment: {env_name} ...")
    env = get_env(
        name=env_name,
        privileged=False,
        enable_render=True,
        viser_debug=False
    )
    
    print("Resetting environment to acquire initial observation...")
    obs, _ = env.reset()
    
    camera_name = "robot0_robotview"
    if camera_name not in obs or "images" not in obs[camera_name]:
        print(f"Error: Could not find image data for camera '{camera_name}'.")
        sys.exit(1)
        
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving images to directory: {os.path.abspath(output_dir)}")
    
    images_dict = obs[camera_name]["images"]
    
    # 1. Save RGB Image as PNG
    if "rgb" in images_dict:
        rgb_data = images_dict["rgb"]
        
        # Ensure the array is of type uint8 before saving as image
        if rgb_data.dtype != np.uint8:
            rgb_data = rgb_data.astype(np.uint8)
            
        img = Image.fromarray(rgb_data)
        rgb_save_path = os.path.join(output_dir, f"{camera_name}_rgb.png")
        img.save(rgb_save_path)
        print(f"  [+] Saved RGB image to: {rgb_save_path} (Shape: {rgb_data.shape})")
    else:
        print("  [-] RGB data not found in observation.")

    # 2. Save Depth Image as NPY
    if "depth" in images_dict:
        depth_data = images_dict["depth"]
        depth_save_path = os.path.join(output_dir, f"{camera_name}_depth.npy")
        np.save(depth_save_path, depth_data)
        print(f"  [+] Saved Depth data to: {depth_save_path} (Shape: {depth_data.shape})")
    else:
        print("  [-] Depth data not found in observation.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage error. Please provide the path to the YAML configuration file.")
        print("Example: python save_images.py env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml")
        sys.exit(1)
        
    config_path = sys.argv[1]
    main(config_path)