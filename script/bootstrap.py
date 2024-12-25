#!/usr/bin/env python3
import platform
import subprocess
import sys
import venv
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e.stderr}", file=sys.stderr)
        sys.exit(1)

def create_venv():
    """Create a virtual environment."""
    venv_dir = Path(".venv")
    if venv_dir.exists():
        print("Virtual environment already exists, skipping creation...")
        return
    
    print("Creating virtual environment...")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(".venv")
    
    # Ensure pip is installed
    python = get_venv_python()
    run_command([python, "-m", "ensurepip", "--upgrade"])
    run_command([python, "-m", "pip", "install", "--upgrade", "pip"])

def get_venv_python():
    """Get the path to the virtual environment's Python executable."""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\python.exe"
    return ".venv/bin/python"

def install_dependencies():
    """Install required packages."""
    print("Installing dependencies...")
    python = get_venv_python()
    run_command([python, "-m", "pip", "install", "-r", "requirements.txt"])

def download_model():
    """Download the TFLite model if no model exists."""
    tflite_files = list(Path(".").glob("*.tflite"))
    if tflite_files:
        print(f"Found existing model: {tflite_files[0]}")
        return
    
    print("No TFLite model found. Downloading default model...")
    model_url = "https://raw.githubusercontent.com/esphome/micro-wake-word-models/main/models/v2/okay_nabu.tflite"
    model_path = Path("okay_nabu.tflite")
    
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Successfully downloaded default model to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)

def check_sox():
    """Check if sox is installed."""
    try:
        run_command(["sox", "--version"], check=False)
        return True
    except FileNotFoundError:
        return False

def main():
    if not check_sox():
        print("\nWARNING: 'sox' is not installed. You'll need to install it to use the microphone:")
        print("- macOS: brew install sox")
        print("- Linux: sudo apt-get install sox")
        print("- Windows: Download from https://sourceforge.net/projects/sox/")
        print("\nContinuing with setup...\n")

    create_venv()
    install_dependencies()
    download_model()
    
    print("Setup complete!")

if __name__ == "__main__":
    main()
