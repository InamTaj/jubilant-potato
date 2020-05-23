# jubilant-potato
Jubilant Potato - a long piece of code potato!

______

# ChexNet-Keras

## Setup

### Docker

2. `docker run --gpus all --rm -it -v ${PWD}:/app/ tensorflow/tensorflow:1.15.2 bash`
3. `cd /app/bruce`
4. `pip install -r requirements.txt`
5. `apt-get install -y libsm6 libxext6 libxrender-dev; pip install opencv-python`
6. `python train.py`
7. `python test.py`
