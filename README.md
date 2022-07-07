# T3-DNN

T3: Tiny Time-series Transformer Deep Neural Network architecture

## Getting Started

Run the tutorial `getting-started.ipydb` using [Jupyter Lab or Jupyter Notebooks](https://jupyter.org/).

### Access Through Remote Docker Container

We like to train our models on a remote server (like lnissrv4.eng.utah.edu). Follow these steps to get it running.

1. Build a docker image for tensorflow on the remote server. Transfer at least the `src` directory over to the remote server, enter it, and run:

```bash
docker build -t tf2.7.0-gpu-jupyter .
```

2. If you want to make full use of a CUDA-enabled GPU, install the [NVIDIA Container Toolkit] (https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) on the server.

3. ssh into the server setting port fowarding so you can access the Jupyter Notebook from your local address

```bash
ssh -L 8888:localhost:8888 <username>@<server>
```

4. Run the container:

```bash
docker run -it --rm -u `id -u `:`id -g` -v `pwd`:/tf -p 8888:8888 tf2.7.0-gpu-jupyter:latest
```

5. Copy the local URL and token from the CLI and paste it into your browser. Open the Getting-Started Notebook
