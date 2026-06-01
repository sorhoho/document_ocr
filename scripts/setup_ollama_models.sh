#!/usr/bin/env bash
# Pull all Ollama models required by this project.
# Run once after installing Ollama: https://ollama.com
set -euo pipefail

echo "==> Pulling Typhoon OCR 1.5 (3B, Q4_K_M, ~3.2 GB)..."
echo "    Note: use Ollama, not raw GGUF — official model card warns GGUF has accuracy issues."
ollama pull scb10x/typhoon-ocr1.5-3b

echo ""
echo "==> Pulling Gemma 3 4B for structured extraction (~3.3 GB)..."
ollama pull gemma3:4b

echo ""
echo "==> All models ready."
ollama list
