import subprocess
import shutil
import os
import glob

CUDA_INC = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include"
CUDA_LIB = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64"

SRC = "src"
DIST = "dist"
BIN = "cuda-nn.exe"
KERNEL = "cudanet"

def run_cmd(cmd):
    print(cmd)
    step = subprocess.run(cmd)
    step.check_returncode()
    
def clear_dir(dir):
    try:
        shutil.rmtree(dir, ignore_errors=True)
    except FileNotFoundError:
        pass
    os.mkdir(dir)

def build_bin():
    cpp = glob.glob(f"{SRC}/**/*.cpp", recursive=True)
    run_cmd(f'g++ -s -O3 "-I{CUDA_INC}" "-L{CUDA_LIB}" {" ".join(cpp)} -o {DIST}/{BIN} -lcuda -lz')

def build_cuda():
    cu = glob.glob(f"{SRC}/**/*.cu", recursive=True)
    run_cmd(f'nvcc --ptx  --generate-line-info --source-in-ptx --output-file {DIST}/{KERNEL}.ptx {" ".join(cu)}')
    
if __name__ == "__main__":
    clear_dir(DIST)
    build_bin()
    build_cuda()
