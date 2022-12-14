FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /home/t-hyun/22f_nlp_zero_shot

COPY ./docker_requirements_no_deps.txt /workspace/requirements_no_deps.txt
COPY ./docker_requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --upgrade pip

RUN pip install  --no-cache-dir --no-deps -r /workspace/requirements_no_deps.txt
RUN pip install  --no-cache-dir -r /workspace/requirements.txt
