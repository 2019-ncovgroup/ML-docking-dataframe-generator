FROM centos:7

LABEL maintainer="wilke@anl.gov"

RUN yum -y upgrade && yum -y update
RUN yum -y install \
    gcc \
    git \
    python3 \
    python3-devel
 






# Dependencies
RUN pip3 install \
    fastparquet \
    joblib \
    matplotlib \
    numpy \
    pandas \
    parquet \
    psutil \
    pyarrow

RUN mkdir /data /features /out

WORKDIR /local/repos/ML-docking-dataframe-generator

# Install from local
COPY . . 
RUN ln -s /data data ; \
    ln -s /features features ; \
    ln -s /out out 

# Install from git
# RUN git clone https://github.com/2019-ncovgroup/ML-docking-dataframe-generator ; \
#     cd ML-docking-dataframe-generator ; \
#     ln -s /data data ; \
#     ln -s /features features ; \
#     ln -s /out out     



WORKDIR /local/repos/ML-docking-dataframe-generator
ENTRYPOINT [ "python3" , "src/main_gen_dfs.py" ]