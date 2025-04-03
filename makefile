create_venv:
	python3 -m venv .venv
	@echo "Virtual environment created in .venv"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

requirements:
	.venv/bin/pip install -r requirements.txt

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  create_venv    Create a virtual environment"
	@echo "  requirements   Install the requirements"
	@echo "  delete_venv    Delete the virtual environment"
	@echo "  help           Show this help message"

delete_venv:
	rm -rf .venv
	@echo "Virtual environment deleted"