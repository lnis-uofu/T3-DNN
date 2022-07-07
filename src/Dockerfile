FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

COPY requirements.txt /tmp/
RUN python3 -m pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt

COPY . /tmp/