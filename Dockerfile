FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

LABEL maintainer="frt frt@hongo.wide.ad.jp"

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
 graphviz


RUN pip install --upgrade setuptools pip
RUN pip install\
 scikit-image\
 opencv-python\
 seaborn\
 matplotlib\
 tqdm\
 sklearn\
 pyarrow\
 timeout_decorator\
 pydot

#INSTALL animeface 
#Copyright (C) 2009-2016 nagadomi <nagadomi@nurs.or.jp>
RUN git clone https://github.com/nagadomi/animeface-2009.git
RUN gem update && gem install rmagick parallel ruby-progressbar progress_bar
RUN cd animeface-2009 && ./build.sh


RUN touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py
#run jupyter without password
#RUN echo c.NotebookApp.password = \'${JUPYTER_PW_HASH}\' >> /root/.jupyter/jupyter_notebook_config.py

#run jupyter without password
#CMD jupyter notebook --allow-root
WORKDIR /
CMD jupyter notebook --allow-root  --NotebookApp.token=''

