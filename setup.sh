#!/bin/sh

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN
