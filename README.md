# jubilant-potato
Jubilant Potato - a long piece of code potato!

______

# ChexNet-Torch

This model is based on https://github.com/ilkarman/DeepLearningFrameworks


## Setup

### Docker

1. `docker pull pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime`
2. `docker run --gpus all --rm -it -v ${PWD}:/app/ pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime bash`
3. `cd /app/`
4. `pip install -r requirements.txt`
5. `python model.py`
