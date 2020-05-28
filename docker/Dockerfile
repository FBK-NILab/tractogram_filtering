FROM continuumio/miniconda3

MAINTAINER Pietro, Ruben and Paolo

ENV DEBIAN_FRONTEND=noninteractive

# basic system stuff [maybe useless]
#RUN apt-get update && apt-get install -y apt-utils git g++ libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev jq strace curl vim wget

## setup conda environment
# the next three lines are maybe not necessary
# RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
# echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
# echo "conda activate base" >> ~/.bashrc

RUN conda install python=3.7
RUN conda install dipy=1.1 -c conda-forge
RUN conda install pytorch=1.5 torchvision=0.6 cpuonly -c pytorch
RUN conda install mrtrix3=3.0 -c mrtrix3
RUN pip install antspyx=0.2.5

# install pytorch geometric https://github.com/rusty1s/pytorch_geometric
ENV CUDA=cpu
RUN pip install torch-scatter==2.0.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-sparse==0.6.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-cluster==1.5.4+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-spline-conv==1.2.0+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-geometric==1.5

# tractogram filtering
RUN git clone https://github.com/FBK-NILab/tractogram-cleaning.git \
    && cd tractogram-cleaning \
    # && git checkout 34980972009516a27e2b99837a4f483ce280cf9a \
    # && pip install . \
    # && cd ..     

#make it work under singularity 
# RUN ldconfig && mkdir -p /N/u /N/home /N/dc2 /N/soft

#https://wiki.ubuntu.com/DashAsBinSh 
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh
