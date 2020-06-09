FROM continuumio/miniconda3

MAINTAINER Pietro, Ruben and Paolo

ENV DEBIAN_FRONTEND=noninteractive

# basic system stuff
RUN apt-get update && apt-get install -y apt-utils git g++ curl vim wget 
# libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev jq strace curl vim wget

## setup conda environment
# the next three lines are maybe not necessary
# RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
# echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
# echo "conda activate base" >> ~/.bashrc

RUN conda install python=3.7 \
    && conda install dipy=1.1 -c conda-forge \
    && conda install pytorch=1.5 torchvision=0.6 cpuonly -c pytorch \
    && conda install mrtrix3=3.0 -c mrtrix3 \
    && pip install antspyx==0.2.4
RUN pip install torchviz
RUN pip install tensorboardx

# install pytorch geometric https://github.com/rusty1s/pytorch_geometric
ENV CUDA=cpu
RUN pip install torch-scatter==2.0.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    && pip install torch-sparse==0.6.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    && pip install torch-cluster==1.5.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    && pip install torch-spline-conv==1.2.0+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    && pip install torch-geometric==1.5

# tractogram filtering
# needed token for private repo
RUN git config --global url."https://d9da97d6c50c116ed374179abd2e7717962971f5:@github.com/".insteadOf "https://github.com/" \
    && git clone https://github.com/FBK-NILab/tractogram-cleaning.git \
    && cd tractogram-cleaning \
    && git checkout docker
  
ENV PATH=/tractogram-cleaning:$PATH

#make it work under singularity 
RUN ldconfig 
# && mkdir -p /N/u /N/home /N/dc2 /N/soft

#https://wiki.ubuntu.com/DashAsBinSh 
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
