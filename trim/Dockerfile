FROM jupyter/base-notebook:python-3.9.2

LABEL maintainer="frt frt@hongo.wide.ad.jp"


RUN pip install --upgrade setuptools pip
RUN pip install\
 opencv-python\
 tqdm\
 timeout_decorator


RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> ~/.jupyter/jupyter_notebook_config.py

COPY trim/trim.ipynb .
CMD jupyter notebook --NotebookApp.token=''

