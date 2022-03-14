FROM tensorflow/tensorflow:2.7.1-gpu

ENV RES101_ZIP_FILE_ID 1WjBgfOUqp4sdRd9BHs4TkdH2EcBtV5ri

RUN pip install --upgrade pip setuptools Cython opencv-python

WORKDIR /
RUN apt update && apt install -y git wget perl build-essential libgl1-mesa-glx curl unzip libopencv-dev
RUN git clone https://github.com/FullteaOfEEIC/anime-face-detector.git
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$RES101_ZIP_FILE_ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$RES101_ZIP_FILE_ID" -O res101_faster_rcnn_iter_60000.zip
RUN unzip res101_faster_rcnn_iter_60000.zip && mv res101_faster_rcnn_iter_60000* anime-face-detector/model


WORKDIR /anime-face-detector
RUN make clean && make

CMD python main.py -i /input -o /output/output.json
