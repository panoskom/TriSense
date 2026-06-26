# TriSense — common tasks. The app runs via Docker; training runs locally.
.PHONY: help install prefetch data train train-gpu evaluate gallery test app up down clean

PYTHON ?= python
DEVICE ?= cpu

help:
	@echo "TriSense make targets:"
	@echo "  make up           - build & run the app + MLflow (docker compose up --build)"
	@echo "  make down         - stop the docker stack"
	@echo "  make install      - create .venv and install training deps (uses uv if present)"
	@echo "  make prefetch     - download the pretrained encoders into the HF cache"
	@echo "  make data         - download RAVDESS and build the subset manifest"
	@echo "  make train        - quick CPU checkpoint (frozen encoders, heads only)"
	@echo "  make train-gpu    - real LoRA run on GPU (see README 'Training')"
	@echo "  make evaluate     - re-evaluate the committed checkpoint -> model card"
	@echo "  make gallery      - rebuild the results gallery from the checkpoint"
	@echo "  make test         - run backend pytest suite"

install:
	uv venv .venv --python 3.12 || python -m venv .venv
	. .venv/bin/activate && (uv pip install -r training/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
		|| pip install -r training/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu)

prefetch:
	$(PYTHON) training/prefetch_models.py

data:
	$(PYTHON) training/download_data.py --actors $(or $(ACTORS),10) --per-class $(or $(PER_CLASS),40)

# Quick throwaway CPU checkpoint so the app runs end-to-end (phase one).
train:
	$(PYTHON) training/train.py --device cpu --epochs 15 --batch-size 16 --lr 1e-3 --run-name phase1-cpu-frozen

# Real run: LoRA on both encoders, GPU, mixed precision (phase two).
# These match the committed checkpoint (84.7% test accuracy on a 20-actor subset).
train-gpu:
	$(PYTHON) training/train.py --device cuda --amp \
		--lora-video --lora-audio --lora-rank 16 --lora-alpha 32 \
		--epochs 15 --batch-size 24 --lr 2e-4 --run-name gpu-lora-both

evaluate:
	$(PYTHON) training/evaluate.py --device $(DEVICE)

gallery:
	$(PYTHON) training/make_gallery.py --device $(DEVICE)

test:
	$(PYTHON) -m pytest backend/tests -v

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf data/subset/cache mlruns frontend/dist
