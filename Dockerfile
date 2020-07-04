FROM nvidia/cuda:10.1-cudnn7-devel

LABEL maintainer="frt frt@hongo.wide.ad.jp"
  
ARG PYTHON_VERSION="3.6.5"
ARG PYTHON_ROOT=/usr/local/bin/python

RUN apt update && apt install -y\
 sudo\
 wget\
 git\
 curl\
 build-essential\
 ruby\
 ffmpeg\
 imagemagick\
 ruby-dev\
 libmagickwand-dev\
 libreadline-dev\
 libncursesw5-dev\
 libssl-dev\
 libsqlite3-dev\
 libgdbm-dev\
 libbz2-dev\
 openjdk-11-jdk\
 liblzma-dev\
 zlib1g-dev\
 uuid-dev\
 libffi-dev\
 libdb-dev\
 libglib2.0-0\
 libsm6\
 libxrender1\
 libxext6\
 graphviz


#INSTALL PYTHON
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && cd ~/.pyenv/plugins/python-build && ./install.sh
RUN /usr/local/bin/python-build -v ${PYTHON_VERSION} ${PYTHON_ROOT}
RUN rm -rf ~/.pyenv
ENV PATH $PATH:$PYTHON_ROOT/bin

RUN pip install --upgrade setuptools pip
RUN pip install\
 numpy\
 tensorflow-gpu==2.1.0\
 scikit-image\
 opencv-python\
 seaborn\
 matplotlib\
 tqdm\
 jupyter\
 pandas\
 sklearn\
 pyarrow\
 timeout_decorator\
 pydot

#INSTALL animeface 
#Copyright (C) 2009-2016 nagadomi <nagadomi@nurs.or.jp>
RUN git clone https://github.com/nagadomi/animeface-2009.git
RUN gem update && gem install rmagick parallel ruby-progressbar progress_bar
RUN cd animeface-2009 && ./build.sh


RUN mkdir /root/.jupyter && touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py
#run jupyter without password
#RUN echo c.NotebookApp.password = \'${JUPYTER_PW_HASH}\' >> /root/.jupyter/jupyter_notebook_config.py

#run jupyter without password
#CMD jupyter notebook --allow-root
CMD jupyter notebook --allow-root  --NotebookApp.token=''

