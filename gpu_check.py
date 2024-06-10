import torch

def check_gpu():
    if torch.cuda.is_available():
        print("CUDA available: True")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA available: False")
        return False

if __name__ == "__main__":
    if not check_gpu():
        exit(1)
