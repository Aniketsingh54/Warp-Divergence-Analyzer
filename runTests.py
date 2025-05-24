import os
import subprocess
import glob

# Configurable paths
BUILD_DIR = "build"
PLUGIN_NAME = "WarpDivergence.so"
TEST_DIR = "test"

def run_test_on_file(test_file):
    cmd = [
        "opt",
        f"-load-pass-plugin", f"./{PLUGIN_NAME}",
        "-passes=warp-divergence",
        "-disable-output",
        test_file
    ]

    print(f"\n=== Running test: {test_file} ===")
    try:
        result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
        print(result.stderr.strip() or result.stdout.strip())
    except Exception as e:
        print(f"Error running test {test_file}: {e}")

def main():
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.ll")))
    if not test_files:
        print("No test .ll files found in 'test/'")
        return

    for test_file in test_files:
        run_test_on_file(os.path.abspath(test_file))

if __name__ == "__main__":
    main()
