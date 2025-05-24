import os
import subprocess
import glob
import argparse

# Configurable paths
BUILD_DIR = "build"
PLUGIN_NAME = "WarpDivergence.so"
TEST_DIR = "test/samples"
CUDA_PATH = "/usr/lib/cuda"  # Make sure this is where your CUDA install is

def build_project():
    print("=== Building the project ===")
    try:
        os.makedirs(BUILD_DIR, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=BUILD_DIR, check=True)
        subprocess.run(["make", "-j"], cwd=BUILD_DIR, check=True)
        print("Build succeeded.\n")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        exit(1)

def compile_cu_to_ll(cu_file):
    ll_file = cu_file[:-3] + ".ll"  # replace .cu with .ll
    print(f"=== Compiling {cu_file} to LLVM IR ===")
    cmd = [
        "clang",
        "--cuda-device-only",
        "-O0",
        "-g",
        cu_file,
        "-S",
        "-emit-llvm",
        "-o", ll_file,
        f"--cuda-path={CUDA_PATH}"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Compiled {cu_file} -> {ll_file}\n")
        return ll_file
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile {cu_file}:\n{e.stderr}")
        return None

def run_test_on_file(test_file):
    print(f"=== Running WarpDivergencePass on function(s) in {test_file} ===")
    cmd = [
        "opt",
        "-load-pass-plugin", os.path.join("..", BUILD_DIR, PLUGIN_NAME),
        "-passes=warp-divergence",
        "-disable-output",
        test_file
    ]
    try:
        result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
        output = result.stderr.strip() or result.stdout.strip()
        if output:
            print(output)
        print()
    except Exception as e:
        print(f"Error running test {test_file}: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Run WarpDivergenceAnalyzer tests.")
    parser.add_argument("-b", "--build", action="store_true", help="Build the project")
    parser.add_argument("-c", "--compile", action="store_true", help="Compile .cu files to .ll")
    args = parser.parse_args()

    if args.build:
        build_project()

    ll_files = []

    if args.compile:
        cu_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.cu")))
        if not cu_files:
            print("No .cu files found in test directory.")
        for cu_file in cu_files:
            ll_file = compile_cu_to_ll(os.path.abspath(cu_file))
            if ll_file:
                ll_files.append(ll_file)

    if not args.compile:
        # Use only existing .ll files if --compile not set
        ll_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.ll")))

    if not ll_files:
        print("No .ll files to run the test on.")
        return

    for ll_file in ll_files:
        run_test_on_file(os.path.abspath(ll_file))

if __name__ == "__main__":
    main()
