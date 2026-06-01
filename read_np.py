import numpy as np

# read npy file and out put its shape and data

def read_npy_file(file_path):
    try:
        data = np.load(file_path)
        print(f"Data shape: {data.shape}")
        print(f"Data:\n{data}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        
if __name__ == "__main__":
    # Replace with your actual file path
    npy_file_path = "trajectory_data/bottle_knock/trajectory_depth_bottleKnock.npy"
    read_npy_file(npy_file_path)