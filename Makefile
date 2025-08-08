# Makefile

# Define variables for commands
PYTEST = pytest
LINTER = flake8

run:
	@echo "Running the main script..."
	python main.py --timestep 6

# Target to run tests
test:
	@echo "Running tests..."
	$(PYTEST) --ff tests

# Target to run linting
lint:
	@echo "Running linter..."
	$(LINTER) .

# Target to prepare a commit
commit:
	@echo "Preparing commit..."
	@read -p "Enter commit message: " m && \
	echo "$$m" > commit_message.txt && \
	git add . && \
	git commit -F commit_message.txt && \
	rm commit_message.txt
	git push

coverage:
	@echo "Running coverage..."
	$(PYTEST) --cov=src --cov-report=json --cov-report=term-missing tests/

# Default target
.PHONY: all
all: test lint
