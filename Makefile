clean:
	find . -type f -name ".DS_Store" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

isort:
	isort .

black:
	black .

flake8:
# Configuration cotrolled by .flake8
	flake8

pylint:
# Configuration cotrolled by .pylintrc
	pylint **/*.py

format: isort black flake8 pylint