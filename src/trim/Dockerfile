FROM python:3.10.3

LABEL maintainer="frt frt@hongo.wide.ad.jp"

RUN apt update && apt install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev

RUN pip install --upgrade setuptools pip
RUN pip install\
 opencv-python\
 tqdm\
 timeout_decorator\
 jupyter\
 matplotlib

RUN mkdir /root/.jupyter
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py

COPY src/trim/*.ipynb /root/
COPY src/trim/*.py /root/
WORKDIR /root
CMD jupyter notebook --allow-root --NotebookApp.token=''

