FROM tensorflow/tensorflow:2.7.1-gpu

RUN pip install --upgrade pip setuptools Cython opencv-python

WORKDIR /
RUN apt update && apt install -y git wget perl build-essential libgl1-mesa-glx curl unzip libopencv-dev
RUN git clone https://github.com/FullteaOfEEIC/anime-face-detector.git
ADD res101_faster_rcnn_iter_60000.zip .
RUN unzip res101_faster_rcnn_iter_60000.zip && mv res101_faster_rcnn_iter_60000* anime-face-detector/model


WORKDIR /anime-face-detector
RUN make clean && make

CMD python main.py -i /input -o /output/output.json
