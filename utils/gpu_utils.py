from tensorflow.config import list_physical_devices
import os

def disable_tensorflow_debugging_logs():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_gpus(gpu_nums=None):
    physical_devices = list_physical_devices('GPU')
    if gpu_nums: physical_devices = physical_devices[:gpu_nums]
    print("Number of GPUs:", len(physical_devices))
    return physical_devices

if __name__ == "__main__":
    disable_tensorflow_debugging_logs()
    devices = get_gpus(2)