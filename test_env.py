# ==============================================================================
# PLACEMENT INSTRUCTIONS:
# Place this script in the ROOT directory of your project. 
# (This is the directory that contains the 'capx' and 'env_configs' folders).
#
# RUN INSTRUCTIONS:
# Execute the script from the terminal, passing the path to your YAML config:
# python test_env.py env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml
# ==============================================================================

import sys
import yaml
import numpy as np

# Import simulators to ensure all low-level environments are registered to _ENV_FACTORIES
import capx.envs.simulators  
from capx.envs.base import get_env

def main(yaml_path: str):
    print(f"Loading config from: {yaml_path}")
    
    # Parse the YAML file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        
    try:
        # Extract the low_level environment name based on the YAML structure
        env_name = config['env']['cfg']['low_level']
    except KeyError as e:
        print(f"Error parsing YAML: Could not find key {e}")
        sys.exit(1)
        
    print(f"Target low_level environment identified: {env_name}")
    print("Initializing environment...")
    
    # Instantiate the low-level simulation environment
    env = get_env(
        name=env_name,
        privileged=False,
        enable_render=True,
        viser_debug=False
    )
    
    print("Resetting environment to acquire initial observation...")
    obs, info = env.reset()
    
    print("\n" + "="*45)
    print("SUCCESS: Initial observation retrieved!")
    print("="*45 + "\n")
    
    print(f"Top-level keys in observation: {list(obs.keys())}")
    
    # Print basic robot state information
    if "robot_joint_pos" in obs:
        print(f"\nRobot joint pos shape: {obs['robot_joint_pos'].shape}")
    if "robot_cartesian_pos" in obs:
        print(f"Robot cartesian pos shape: {obs['robot_cartesian_pos'].shape}")
        
    # Print default camera information
    camera_name = "robot0_robotview"
    if camera_name in obs:
        print(f"\nCamera [{camera_name}] data structure:")
        cam_data = obs[camera_name]
        for key, value in cam_data.items():
            if isinstance(value, np.ndarray):
                print(f"  ├── {key}: np.ndarray, shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, dict):
                print(f"  ├── {key}: dict with keys {list(value.keys())}")
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, np.ndarray):
                        print(f"  │     └── {sub_k}: shape {sub_v.shape}")
            else:
                print(f"  ├── {key}: {type(value)}")
                
    return obs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage error. Please provide the path to the YAML configuration file.")
        print("Example: python test_env.py env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml")
        sys.exit(1)
        
    config_path = sys.argv[1]
    initial_obs = main(config_path)