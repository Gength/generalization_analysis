#!/bin/bash
# M5: AVATAR (RelGAN-based generalization) via Docker
# Docker image: avatar-tf1 (built from benchmark/docker/Dockerfile.avatar)
# Full run takes 2-6 hours. Use --quick for demo (~30 min).
uv run python benchmark/docker/run_avatar.py "$@"
