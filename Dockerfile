from anibali/pytorch:2.0.0-cuda11.8

# replace resource
RUN sudo sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN sudo sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN sudo apt-get update && sudo apt-get install -y gcc

WORKDIR /app/lavis
USER user:user

COPY --chown=user:user . .

RUN pip install -e .
