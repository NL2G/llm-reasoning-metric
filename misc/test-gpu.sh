#!/usr/bin/env bash

srun --pty --gres=gpu:1 --mem=10G --time=01:00:00 --cpus-per-task=4 --partition=gpu-vram-94gb python -c "import torch; print(torch.cuda.is_available());"
