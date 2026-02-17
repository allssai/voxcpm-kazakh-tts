FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装 Python 和必要的系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip3 install --upgrade pip

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p voices lora

# 暴露端口
EXPOSE 7870

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7870

# 启动命令
CMD ["python3", "web_app.py"]
