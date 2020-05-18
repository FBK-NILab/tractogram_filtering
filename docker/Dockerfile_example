FROM neurodebian:stretch-non-free

MAINTAINER Pietro and Paolo (from Soichi Hayashis <hayashis@iu.edu>)

ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get update && apt-get install -y git g++ python python-numpy libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev fsl-complete python-pip jq strace curl vim 
RUN apt-get update && apt-get install -y apt-utils git g++ libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev fsl-complete jq strace curl vim wget

## install conda
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda install numpy

#libgomp1 seems to comes with pytorch so I don't need it

#RUN pip install --upgrade pip

## install and compile mrtrix3
#RUN git clone https://github.com/MRtrix3/mrtrix3.git
#RUN cd mrtrix3 && git fetch --tags && git checkout tags/3.0_RC3 && ./configure && ./build
#ENV PATH=$PATH:/mrtrix3/bin

RUN mkdir -p ~/.tractseg \
    && mkdir -p /code \
    && curl -SL -o /code/mrtrix3_RC3.tar.gz \
       https://zenodo.org/record/1415322/files/mrtrix3_RC3.tar.gz?download=1
RUN tar -zxvf /code/mrtrix3_RC3.tar.gz -C code \
    && /code/mrtrix3/set_path

#RUN pip install seaborn
#RUN pip install torch torchvision
RUN conda install -c anaconda seaborn

RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

#install batchgenerator/tractseg
#RUN pip install https://github.com/MIC-DKFZ/batchgenerators/archive/master.zip \
#    && pip install https://github.com/MIC-DKFZ/TractSeg/archive/v1.7.1.zip
RUN git clone https://github.com/MIC-DKFZ/batchgenerators.git \
    && cd batchgenerators \
    && git checkout 34980972009516a27e2b99837a4f483ce280cf9a \
    && pip install . \
    && cd ..
    
RUN git clone https://github.com/FBK-NILab/TractSeg-BrainLife.git \
    && cd TractSeg-BrainLife \
    && git checkout brainlife-app \
    && pip install . \
    && cd ..
     
#RUN HOME=/ download_all_pretrained_weights

#make it work under singularity 
RUN ldconfig && mkdir -p /N/u /N/home /N/dc2 /N/soft

#https://wiki.ubuntu.com/DashAsBinSh 
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
