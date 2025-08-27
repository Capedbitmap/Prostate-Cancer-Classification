.PHONY: help install install-dev clean lint format test run setup-env

# Default target
help:
	@echo "Available commands:"
	@echo "  setup-env    - Create virtual environment and install dependencies"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  clean        - Clean up generated files"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  test         - Run tests"
	@echo "  run          - Run the main training pipeline"
	@echo "  help         - Show this help message"

# Setup complete development environment
setup-env:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Virtual environment created! Activate with: source venv/bin/activate"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install black flake8 pytest jupyter

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf SICAPv2/
	rm -rf SICAPv2_results/
	rm -f *.zip

# Run linting
lint:
	flake8 src/ main.py --max-line-length=88 --extend-ignore=E203,W503

# Format code
format:
	black src/ main.py --line-length=88

# Run tests (when test files are created)
test:
	pytest tests/ -v

# Run the main training pipeline
run:
	python main.py

# Initialize git repository (if not already done)
git-init:
	git init
	git add .
	git commit -m "Initial commit: Project setup"
