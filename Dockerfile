# 使用 NVIDIA CUDA 12.6 Base 版本
FROM nvidia/cuda:12.6.0-base-ubuntu24.04

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# 關閉 PEP 668 保護機制， 使 pip 能 install
ENV PIP_BREAK_SYSTEM_PACKAGES=1


# 安裝基礎工具 (Ubuntu 24.04 內建 python3 就是 3.12)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 連結 Python
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir -p /app/shared_data
WORKDIR /app

# 升級 Pip
#RUN python -m pip install --no-cache-dir --upgrade pip

# 安裝 PyTorch (建議使用 cu126 以獲得最佳相容性)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --break-system-packages


# 2. 將本地的 requirements.txt 複製到鏡像內的 /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages


# 驗證安裝結果
CMD ["python", "-c", "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"]