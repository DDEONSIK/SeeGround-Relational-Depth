# ==================================================================
# 스테이지 1: 빌더 환경 (Builder Environment)
# ==================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 빌드에 필요한 모든 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --no-cache-dir --upgrade pip

# 가상 환경 생성 및 활성화
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# PyTorch 및 의존성 설치
RUN python -m pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ========================== 수정 지점 ==========================
# 1. PyTorch3D 빌드에 필요한 NVIDIA CUB 라이브러리 설치
RUN git clone https://github.com/NVIDIA/cub.git /cub
ENV CUB_HOME=/cub

# 2. 빌드 환경에 타겟 GPU 아키텍처(RTX 3090 -> 8.6)를 명시적으로 지정
ENV TORCH_CUDA_ARCH_LIST="8.6"
# =============================================================

# PyTorch3D 소스에서 컴파일
RUN git clone https://github.com/facebookresearch/pytorch3d.git /pytorch3d
WORKDIR /pytorch3d
# FORCE_CUDA=1 환경 변수는 유지하여 GPU 지원 컴파일을 강제
RUN FORCE_CUDA=1 python -m pip install --no-cache-dir .
WORKDIR /

# ==================================================================
# 스테이지 2: 최종 런타임 환경 (Final Runtime Environment)
# ==================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 실행에 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Builder에서 완성된 가상 환경을 그대로 복사
ENV VENV_PATH=/opt/venv
COPY --from=builder $VENV_PATH $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# 작업 디렉토리 설정 및 코드 복사
WORKDIR /app
COPY . .

CMD ["/bin/bash"]

