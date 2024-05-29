FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av xformers==0.0.25 runpod \
    einops open_clip_torch pytorch_lightning==1.7.7 omegaconf torchmetrics==0.11.4 transformers diffusers==0.27.2 \
    git+https://github.com/camenduru/BasicSR@dev

RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/camenduru/ControlNet-v1-1-nightly /content/ControlNet-v1-1-nightly

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fix_fp16.safetensors -d /content/ControlNet-v1-1-nightly/models -o anything-v3-full.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth -d /content/ControlNet-v1-1-nightly/models -o control_v11p_sd15s2_lineart_anime.pth

COPY ./worker_runpod.py /content/ControlNet-v1-1-nightly/worker_runpod.py
WORKDIR /content/ControlNet-v1-1-nightly
CMD python worker_runpod.py
