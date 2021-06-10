FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt update && apt install -y\
 sudo\
 wget\
 git\
 curl
 
RUN git clone https://github.com/NVlabs/stylegan2.git
 
RUN pip install --upgrade setuptools pip\
 scipy==1.3.3\
 requests==2.22.0\
 Pillow==6.2.1


