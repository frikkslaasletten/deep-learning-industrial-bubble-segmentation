FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt update && apt install -y \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create Python 3.12 environment
RUN conda create -y -n sam3 python=3.12

# Activate environment for following RUN commands
SHELL ["conda", "run", "-n", "sam3", "/bin/bash", "-c"]

# Upgrade pip
RUN pip install --upgrade pip

# Install python dependencies 
RUN pip install einops decord opencv-python pycocotools psutil 
RUN pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Set working directory
WORKDIR /workspace

# Clone SAM3
RUN git clone https://github.com/facebookresearch/sam3.git
WORKDIR /workspace/sam3
RUN git checkout 86ed77094094e5cabb16b0414ec60c5ba9ce0a0f

# Make directories
RUN mkdir -p results/{original,mask,overlay}
RUN mkdir input_vids

# Copy scripts and videos
COPY vid_to_images.py /workspace/sam3/
COPY main.py /workspace/sam3/
COPY input_vids/. /workspace/sam3/input_vids
COPY readme.txt /workspace/sam3/

# Install dependencies
RUN pip install -e .

CMD ["/bin/bash"]
