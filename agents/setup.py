#!/usr/bin/env python3
"""
Setup script for the Agentic Marketplace AI Agents.

This script helps initialize the development environment and install dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    return subprocess.run(command, shell=True, check=check, capture_output=True, text=True)


def check_python_version():
    """Check if Python version is compatible (3.9+)."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")


def check_dependencies():
    """Check if required system dependencies are available."""
    dependencies = ["git", "curl"]
    
    for dep in dependencies:
        if not shutil.which(dep):
            print(f"❌ {dep} is not installed or not in PATH")
            sys.exit(1)
        print(f"✅ {dep} is available")


def create_virtual_environment():
    """Create a Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return
    
    print("Creating virtual environment...")
    run_command(f"{sys.executable} -m venv venv")
    print("✅ Virtual environment created")


def activate_and_install():
    """Install Python dependencies in the virtual environment."""
    print("Installing Python dependencies...")
    
    # Determine the activation script based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = r"venv\Scripts\pip"
        python_cmd = r"venv\Scripts\python"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip")
    
    # Install requirements
    run_command(f"{pip_cmd} install -r requirements.txt")
    
    print("✅ Dependencies installed")


def setup_environment_file():
    """Set up the environment configuration file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your API keys and configuration")
    else:
        print("❌ .env.example file not found")


def setup_pre_commit():
    """Set up pre-commit hooks for code quality."""
    try:
        # Install pre-commit if not already installed
        run_command("venv/bin/pip install pre-commit")
        
        # Create a basic pre-commit config if it doesn't exist
        precommit_config = Path(".pre-commit-config.yaml")
        if not precommit_config.exists():
            config_content = """
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 7.1.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
"""
            precommit_config.write_text(config_content.strip())
        
        # Install pre-commit hooks
        run_command("venv/bin/pre-commit install")
        print("✅ Pre-commit hooks installed")
        
    except subprocess.CalledProcessError:
        print("⚠️  Could not set up pre-commit hooks (optional)")


def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "data",
        "outputs",
        "tests/unit",
        "tests/integration"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Edit the .env file and add your API keys:")
    print("   - OpenAI API key (or Anthropic)")
    print("   - Blockchain RPC URLs")
    print("   - Private keys for testing")
    print("\n2. Activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("   .\\venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("\n3. Run tests to verify setup:")
    print("   pytest tests/")
    
    print("\n4. Start developing your agents:")
    print("   python -m agents.service_providers.data_analysis_agent")
    
    print("\n5. Check the documentation:")
    print("   - README.md for project overview")
    print("   - docs/ folder for detailed documentation")
    
    print("\n📝 Remember to:")
    print("   - Keep your .env file secure and never commit it")
    print("   - Update requirements.txt if you add new dependencies")
    print("   - Run tests before committing changes")


def main():
    """Main setup function."""
    print("🚀 Setting up Agentic Marketplace AI Agents")
    print("="*50)
    
    # Change to the agents directory
    os.chdir(Path(__file__).parent)
    
    # Run setup steps
    check_python_version()
    check_dependencies()
    create_virtual_environment()
    activate_and_install()
    setup_environment_file()
    create_directories()
    setup_pre_commit()
    
    display_next_steps()


if __name__ == "__main__":
    main()