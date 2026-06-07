# ==============================================================================
# PLACEMENT INSTRUCTIONS:
# Place this script in the ROOT directory of your project.
# (This is the directory containing the 'capx' and 'env_configs' folders).
#
# RUN INSTRUCTIONS:
# Execute the script from the terminal, passing the path to your YAML config:
# python save_all_cameras.py env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml
# ==============================================================================

import os
import sys
import yaml
import numpy as np
from PIL import Image
from robosuite.utils.camera_utils import get_real_depth_map

# Import simulators to ensure all low-level environments are registered
import capx.envs.simulators  
from capx.envs.base import get_env

def save_rgb_and_depth(rgb_data, depth_data, camera_name, output_dir):
    # 1. Save RGB Image as PNG
    if rgb_data.dtype != np.uint8:
        rgb_data = rgb_data.astype(np.uint8)
    img = Image.fromarray(rgb_data)
    rgb_save_path = os.path.join(output_dir, f"{camera_name}_rgb.png")
    img.save(rgb_save_path)
    print(f"  [+] Saved RGB image to: {rgb_save_path} (Shape: {rgb_data.shape})")

    # 2. Save Depth Image as NPY
    depth_save_path = os.path.join(output_dir, f"{camera_name}_depth.npy")
    np.save(depth_save_path, depth_data)
    print(f"  [+] Saved Depth data to: {depth_save_path} (Shape: {depth_data.shape})")

def main(yaml_path: str, output_dir: str = "output_images_flip"):
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
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving images to directory: {os.path.abspath(output_dir)}")
    
    # ==========================================================================
    # 1. Process Default Camera (robot0_robotview) from Observation
    # ==========================================================================
    main_cam = "robot0_robotview"
    if main_cam in obs and "images" in obs[main_cam]:
        print(f"\nProcessing camera: {main_cam} (From observation)")
        images_dict = obs[main_cam]["images"]
        save_rgb_and_depth(images_dict["rgb"], images_dict["depth"], main_cam, output_dir)
    else:
        print(f"[-] Main camera '{main_cam}' data not found in observation.")

    # ==========================================================================
    # 2. Process Wrist Camera (robot0_eye_in_hand) via Direct Simulation Render
    # ==========================================================================
    wrist_cam = "robot0_eye_in_hand"
    print(f"\nProcessing camera: {wrist_cam} (Direct rendering from simulation context)")
    
    try:
        # Render both RGB and Depth from the simulation context for the wrist camera
        # Sim render returns raw arrays where the vertical axis is inverted (flipped)
        rgb_raw, depth_raw = env.robosuite_env.sim.render(
            camera_name=wrist_cam,
            width=env._render_width,
            height=env._render_height,
            depth=True
        )
        
        # Invert vertically to align with correct image coordinates
        rgb_rectified = rgb_raw[::-1]
        depth_inverted = depth_raw[::-1]
        
        # Convert raw depth values into standard metric depth map (meters)
        depth_metric = get_real_depth_map(env.robosuite_env.sim, depth_inverted)
        
        save_rgb_and_depth(rgb_rectified, depth_metric, wrist_cam, output_dir)
        
    except Exception as e:
        print(f"[-] Failed to render wrist camera '{wrist_cam}': {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage error. Please provide the path to the YAML configuration file.")
        print("Example: python save_all_cameras.py env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml")
        sys.exit(1)
        
    config_path = sys.argv[1]
    main(config_path)