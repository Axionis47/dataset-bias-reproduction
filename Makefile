# Dataset Bias Reproduction - Makefile
# Laptop-friendly reproduction of "A Decade's Battle on Dataset Bias: Are We There Yet?"

.PHONY: help setup download sample train eval corrupt_eval corrupt_report report smoke clean test lint

# Default values
COMBO ?= Combo-1
CFG ?= experiments/$(COMBO).yaml
CKPT ?= experiments/$(COMBO)/best.ckpt
N ?= 10000
MODEL ?= convnext_tiny
EPOCHS ?= 30

help: ## Show this help message
	@echo "Dataset Bias Reproduction - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make setup                              # Set up environment"
	@echo "  make download                           # Download all datasets"
	@echo "  make train combo=Combo-1                # Train Combo-1 model"
	@echo "  make eval combo=Combo-1                 # Evaluate Combo-1 model"
	@echo "  make corrupt_eval combo=Combo-1         # Run corruption evaluation"
	@echo "  make smoke                              # Quick end-to-end test"

setup: ## Create virtual environment and install dependencies
	@echo "🚀 Setting up environment for Mac M4..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip freeze > requirements-lock.txt
	@echo "✅ Verifying MPS availability..."
	./venv/bin/python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ MPS available: {torch.backends.mps.is_available()}'); print(f'✅ MPS built: {torch.backends.mps.is_built()}')"
	@echo "✅ Environment setup complete!"

download: ## Download all datasets with resumable downloads
	@echo "📥 Starting dataset downloads..."
	./venv/bin/python scripts/download_yfcc.py
	./venv/bin/python scripts/download_cc.py
	./venv/bin/python scripts/download_datacomp.py
	./venv/bin/python scripts/download_wit.py
	./venv/bin/python scripts/download_laion.py
	./venv/bin/python scripts/download_imagenet.py
	@echo "✅ Download phase complete!"

sample: ## Create balanced samples and train/val splits
	@echo "🎯 Sampling datasets (N=$(N) per dataset)..."
	./venv/bin/python -m src.data.sampler --n-per-dataset $(N) --val-size 1000
	@echo "✅ Sampling complete!"

train: ## Train model on specified combination
	@echo "🏋️ Training $(COMBO) with $(MODEL) for $(EPOCHS) epochs..."
	./venv/bin/python -m src.train.trainer --config $(CFG) --model $(MODEL) --epochs $(EPOCHS)
	@echo "✅ Training complete!"

eval: ## Evaluate trained model
	@echo "📊 Evaluating $(COMBO)..."
	./venv/bin/python -m src.eval.evaluator --config $(CFG) --checkpoint $(CKPT)
	@echo "✅ Evaluation complete!"

corrupt_eval: ## Run corruption robustness evaluation
	@echo "🌪️ Running corruption robustness evaluation..."
	./venv/bin/python -m src.eval.corruptions --combo $(COMBO) --checkpoint $(CKPT)
	@echo "✅ Corruption evaluation complete!"

corrupt_report: ## Generate corruption robustness report section
	@echo "📝 Generating corruption robustness report..."
	./venv/bin/python scripts/generate_corruption_report.py --combo $(COMBO)
	@echo "✅ Corruption report generated!"

report: ## Generate comprehensive report in Indian English
	@echo "📋 Generating comprehensive report..."
	./venv/bin/python scripts/generate_report.py
	@echo "✅ Report generated: reports/REPORT.md"

smoke: ## Quick 3-minute end-to-end smoke test
	@echo "💨 Running smoke test (3-minute end-to-end verification)..."
	./venv/bin/python scripts/smoke_test.py
	@echo "✅ Smoke test complete!"

test: ## Run all tests
	@echo "🧪 Running tests..."
	./venv/bin/python -m pytest tests/ -v
	@echo "✅ Tests complete!"

lint: ## Run code formatting and linting
	@echo "🧹 Running code formatting..."
	./venv/bin/python -m black src/ scripts/ tests/
	./venv/bin/python -m isort src/ scripts/ tests/
	./venv/bin/python -m flake8 src/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "✅ Linting complete!"

clean: ## Clean up generated files and caches
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Experiment shortcuts
combo1: ## Run Combo-1 (YFCC + CC + DataComp)
	$(MAKE) train combo=Combo-1
	$(MAKE) eval combo=Combo-1

combo2: ## Run Combo-2 (YFCC + CC + DataComp + WIT)
	$(MAKE) train combo=Combo-2
	$(MAKE) eval combo=Combo-2

all6: ## Run All-6 (all datasets)
	$(MAKE) train combo=All-6
	$(MAKE) eval combo=All-6

# Development shortcuts
dev-setup: setup ## Development setup with additional tools
	./venv/bin/pip install pre-commit jupyter notebook
	./venv/bin/pre-commit install

notebook: ## Start Jupyter notebook server
	./venv/bin/jupyter notebook notebooks/

# CI/CD helpers
ci-test: ## Run tests in CI environment
	python -m pytest tests/ -v --tb=short

ci-lint: ## Run linting in CI environment
	python -m black --check src/ scripts/ tests/
	python -m isort --check-only src/ scripts/ tests/
	python -m flake8 src/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503

# Status and info
status: ## Show current experiment status
	@echo "📊 Current Status:"
	@echo "  Repository: $(shell git remote get-url origin 2>/dev/null || echo 'No remote')"
	@echo "  Branch: $(shell git branch --show-current 2>/dev/null || echo 'No git')"
	@echo "  Python: $(shell ./venv/bin/python --version 2>/dev/null || echo 'No venv')"
	@echo "  PyTorch: $(shell ./venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "  MPS: $(shell ./venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo 'Unknown')"
	@echo "  Disk usage: $(shell du -sh data/ 2>/dev/null || echo 'No data/')"
	@echo "  Experiments: $(shell ls experiments/ 2>/dev/null | wc -l | tr -d ' ') configs"

info: ## Show detailed system information
	@echo "💻 System Information:"
	@echo "  OS: $(shell uname -s) $(shell uname -r)"
	@echo "  Architecture: $(shell uname -m)"
	@echo "  CPU: $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
	@echo "  Memory: $(shell echo $$(($$(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024))GB"
	@echo "  Python: $(shell python3 --version)"
	@echo "  Git: $(shell git --version)"
	@echo "  Available space: $(shell df -h . | tail -1 | awk '{print $$4}')"
