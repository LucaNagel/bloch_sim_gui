#!/usr/bin/env python
"""
build.py - Build script for the Bloch simulator

This script handles the compilation of the Cython extension
and sets up the environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required = ['numpy', 'cython']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} found")
        except ImportError:
            print(f"✗ {package} not found")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Please install them with: pip install " + ' '.join(missing))
        return False
    
    return True


def check_compiler():
    """Check if a C compiler is available."""
    print("\nChecking compiler...")
    
    system = platform.system()
    
    if system == "Windows":
        # Check for MSVC
        try:
            subprocess.run(["cl"], capture_output=True)
            print("✓ MSVC compiler found")
            return True
        except FileNotFoundError:
            print("✗ MSVC compiler not found")
            print("Please install Visual Studio Build Tools")
            return False
            
    else:
        # Check for gcc/clang
        for compiler in ['gcc', 'clang']:
            try:
                subprocess.run([compiler, "--version"], capture_output=True)
                print(f"✓ {compiler} compiler found")
                return True
            except FileNotFoundError:
                continue
        
        print("✗ No C compiler found")
        print("Please install gcc or clang")
        return False


def check_openmp():
    """Check if OpenMP is available."""
    print("\nChecking OpenMP support...")
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Check if libomp is installed
        brew_path = "/usr/local/opt/libomp"
        homebrew_arm_path = "/opt/homebrew/opt/libomp"
        
        if Path(brew_path).exists() or Path(homebrew_arm_path).exists():
            print("✓ OpenMP (libomp) found")
            return True
        else:
            print("⚠ OpenMP not found on macOS")
            print("Install with: brew install libomp")
            print("Continuing without OpenMP (single-threaded mode)")
            return False
    
    # For Linux/Windows, OpenMP usually comes with the compiler
    return True


def build_extension():
    """Build the Cython extension."""
    print("\nBuilding Cython extension...")
    
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Extension built successfully")
            
            # Check if the .so/.pyd file was created
            import glob
            extensions = glob.glob("bloch_simulator_cy*.so") + glob.glob("bloch_simulator_cy*.pyd")
            if extensions:
                print(f"✓ Extension file created: {extensions[0]}")
            return True
        else:
            print("✗ Build failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Build failed: {e}")
        return False


def test_import():
    """Test if the extension can be imported."""
    print("\nTesting import...")
    
    try:
        import bloch_simulator_cy
        print("✓ Extension imported successfully")
        
        # Check available functions
        functions = [name for name in dir(bloch_simulator_cy) if not name.startswith('_')]
        print(f"Available functions: {', '.join(functions[:5])}...")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Requirements installed")
        return True
    else:
        print("✗ Failed to install requirements")
        print("You can install manually with: pip install -r requirements.txt")
        return False


def main():
    """Main build process."""
    print("="*60)
    print("Bloch Simulator Build Script")
    print("="*60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check environment
    if not check_dependencies():
        print("\nWould you like to install missing dependencies? (y/n): ", end="")
        if input().lower() == 'y':
            install_requirements()
        else:
            print("Please install dependencies manually and run again.")
            return 1
    
    if not check_compiler():
        return 1
    
    has_openmp = check_openmp()
    
    # Build extension
    if not build_extension():
        print("\nBuild failed. Please check the error messages above.")
        return 1
    
    # Test import
    if not test_import():
        print("\nImport test failed. The extension may not be properly built.")
        return 1
    
    print("\n" + "="*60)
    print("✓ Build completed successfully!")
    print("="*60)
    
    print("\nYou can now:")
    print("1. Run the GUI: python bloch_gui.py")
    print("2. Use the API: from bloch_simulator import BlochSimulator")
    print("3. Run examples: python -c 'from bloch_simulator import example_fid; example_fid()'")
    
    if not has_openmp:
        print("\nNote: OpenMP support is not available. The simulator will run")
        print("in single-threaded mode. For better performance, install OpenMP.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
