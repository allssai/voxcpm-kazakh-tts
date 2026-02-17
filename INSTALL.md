# 安装指南

## 📋 部署说明

本项目需要在本地构建 Python 环境。用户需要：

1. **克隆代码** - 从 GitHub 下载项目代码
2. **下载模型** - 下载 LoRA 微调模型（~31 MB）
3. **安装依赖** - 安装 Python 依赖包（~2 GB）
4. **首次启动** - 自动下载基础模型（~1.5 GB）

总共需要约 **4 GB** 磁盘空间和 **10-20 分钟**安装时间。

---

## 📋 系统要求

### 最低配置
- Python 3.8+
- 8 GB RAM
- 4 GB 磁盘空间
- 网络连接（用于下载模型）

### 推荐配置
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 10 GB 磁盘空间

---

## 🚀 完整安装步骤

### 步骤 1: 克隆项目

```bash
git clone https://github.com/allssai/voxcpm-kazakh-tts
cd voxcpm-kazakh-tts
```

### 步骤 2: 下载 LoRA 模型

LoRA 模型托管在 HuggingFace Hub，需要单独下载：

**方式 1: 使用 huggingface-cli（推荐）**

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载 LoRA 模型到 ./lora/ 目录
huggingface-cli download ErnarBahat/VoxCPM-KazakhTTS-Lora --local-dir ./lora
```

**方式 2: 手动下载**

1. 访问 https://huggingface.co/ErnarBahat/VoxCPM-KazakhTTS-Lora
2. 下载以下文件到 `./lora/` 目录：
   - `lora_weights.safetensors` (~31 MB)
   - `lora_config.json`
   - `optimizer.pth`
   - `scheduler.pth`
   - `voxcpm_finetune_lora.yaml`

### 步骤 3: 安装依赖

**自动安装（推荐）**

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```bash
install.bat
```

安装脚本会自动：
- 检查 Python 版本
- 创建虚拟环境（可选）
- 安装所有依赖包
- 验证 CUDA 支持

**手动安装**

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 步骤 4: 启动应用

**Linux/macOS:**
```bash
./start.sh
```

**Windows:**
```bash
start.bat
```

**Python 直接启动:**
```bash
python web_app.py
```

### 步骤 5: 访问应用

打开浏览器访问: http://localhost:7860

> **首次启动说明**: 
> - 首次运行会自动从 HuggingFace 下载 VoxCPM 基础模型（~1.5 GB）
> - 下载时间取决于网络速度，通常需要 5-15 分钟
> - 模型会缓存到 `~/.cache/huggingface/`，后续启动只需 10 秒
> - 如果下载失败，可以使用镜像站点（见下方常见问题）

---

## 🐳 Docker 部署（可选）

如果你熟悉 Docker，可以使用容器化部署：

### 前提条件
- Docker 20.10+
- Docker Compose 1.29+
- NVIDIA Docker（用于 GPU 支持）

### 部署步骤

1. **克隆项目并下载 LoRA**（同上）

2. **启动容器**
```bash
docker-compose up -d
```

3. **查看日志**
```bash
docker-compose logs -f
```

4. **访问应用**
http://localhost:7860

5. **停止容器**
```bash
docker-compose down
```

---

## ⚙️ 配置选项

### 离线模式

如果需要完全离线运行（首次需要在线下载模型）：

```bash
export HF_HUB_OFFLINE=1  # Linux/macOS
set HF_HUB_OFFLINE=1     # Windows
```

### 自定义端口

修改 `web_app.py` 最后一行：

```python
demo.launch(server_name="0.0.0.0", server_port=7860)  # 改为其他端口
```

### GPU/CPU 模式

系统会自动检测 CUDA 并使用 GPU。如果没有 GPU，会自动使用 CPU 模式（速度较慢）。

---

## 🔧 常见问题

### Q: 为什么需要下载这么多东西？

**A**: 项目包含三部分：
1. **代码**（~10 MB）- 从 GitHub 克隆
2. **LoRA 模型**（~31 MB）- 从 HuggingFace 下载
3. **基础模型**（~1.5 GB）- 首次启动自动下载
4. **Python 依赖**（~2 GB）- pip 安装

总共约 4 GB，这是深度学习项目的正常大小。

### Q: 可以不下载模型吗？

**A**: 不可以。模型是 TTS 的核心，必须下载。但模型会缓存到本地，只需下载一次。

### Q: pip 安装失败

**A**: 使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 模型下载失败

**A**: 使用 HuggingFace 镜像站点：

```bash
# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows
set HF_ENDPOINT=https://hf-mirror.com

# 然后重新启动
```

### Q: CUDA 不可用

**A**: 检查 CUDA 安装：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果没有 GPU，系统会自动使用 CPU 模式（速度较慢但可用）。

### Q: 端口被占用

**A**: 修改端口号或停止占用端口的进程：

```bash
# 查找占用端口的进程
lsof -i :7860          # Linux/macOS
netstat -ano | findstr :7860  # Windows

# 杀死进程
kill -9 <PID>          # Linux/macOS
taskkill /PID <PID> /F # Windows
```

### Q: 虚拟环境是必须的吗？

**A**: 不是必须的，但强烈推荐。虚拟环境可以：
- 避免依赖冲突
- 保持系统 Python 环境干净
- 方便卸载和重装

---

## 📦 文件说明

### 需要下载的文件

| 文件 | 大小 | 来源 | 说明 |
|------|------|------|------|
| 项目代码 | ~10 MB | GitHub | 克隆仓库 |
| LoRA 模型 | ~31 MB | HuggingFace | 手动下载 |
| 基础模型 | ~1.5 GB | HuggingFace | 自动下载 |
| Python 依赖 | ~2 GB | PyPI | pip 安装 |

### 目录结构

```
voxcpm-kazakh-tts/
├── lora/                 # LoRA 模型（需要下载）
│   ├── lora_weights.safetensors
│   └── ...
├── voices/               # 音色预设（已包含）
├── voxcpm/               # 核心代码（已包含）
├── web_app.py            # 主程序（已包含）
└── requirements.txt      # 依赖列表（已包含）
```

---

## 📚 下一步

安装完成后，请查看：

- [README.md](README.md) - 项目文档和使用说明
- [docs/音色管理完整指南.md](docs/音色管理完整指南.md) - 音色管理
- [docs/音色相似度优化指南.md](docs/音色相似度优化指南.md) - 音色优化

---

## 💡 部署建议

### 个人使用
- 使用自动安装脚本
- 创建虚拟环境
- 首次启动耐心等待模型下载

### 服务器部署
- 使用 Docker 部署
- 配置 Nginx 反向代理
- 使用 systemd 管理服务

### 离线环境
1. 在有网络的机器上完成所有下载
2. 打包整个项目目录
3. 复制到离线机器
4. 设置 `HF_HUB_OFFLINE=1`

---

**祝你使用愉快！** 🎉
