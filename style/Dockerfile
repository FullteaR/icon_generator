FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel


RUN apt update && apt upgrade -y && apt install -y git
RUN pip install --upgrade setuptools pip imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 click scipy Ninja

RUN git clone https://github.com/NVlabs/stylegan3.git /stylegan3

WORKDIR /stylegan3

