# 多语言 TTS 引擎 - 哈萨克语强化版

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)

基于 VoxCPM 1.5 的多语言语音合成系统，专门针对哈萨克语进行了 LoRA 微调优化。

[快速开始](#-快速开始) • [文档](INSTALL.md)

</div>

---

## ✨ 主要特性

- 🌍 **多语言支持**: 哈萨克语、中文、英文及混合文本
- 🎭 **零样本音色克隆**: 3-10秒参考音频即可克隆音色
- ⚡ **实时合成**: 快速生成高质量语音
- 🎯 **哈萨克语优化**: 通过 LoRA 微调提升哈萨克语合成质量
- 🎨 **双语界面**: 中文和哈萨克语完整支持
- 📊 **实时进度**: 显示推理过程和详细日志
- 🔧 **灵活配置**: 支持语速、音调、推理步数等参数调整
- 🎵 **音色管理**: 完整的音色增删改查功能

## 🚀 快速开始

### 前提条件

- Python 3.8+ 
- 8GB+ RAM
- 4GB+ 磁盘空间
- （推荐）NVIDIA GPU with CUDA

### 三步部署

#### 1. 克隆项目

```bash
git clone https://github.com/allssai/voxcpm-kazakh-tts
cd voxcpm-kazakh-tts
```

#### 2. 下载 LoRA 模型

LoRA 模型托管在 HuggingFace，需要单独下载：

```bash
# 方式 1: 使用 huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download ErnarBahat/VoxCPM-KazakhTTS-Lora --local-dir ./lora

# 方式 2: 手动下载
# 访问 https://huggingface.co/ErnarBahat/VoxCPM-KazakhTTS-Lora
# 下载所有文件到 ./lora/ 目录
```

#### 3. 安装并启动

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh    # 自动安装依赖
./start.sh      # 启动应用
```

**Windows:**
```bash
install.bat     # 自动安装依赖
start.bat       # 启动应用
```

**Docker:**
```bash
docker-compose up -d
```

#### 4. 访问应用

打开浏览器访问: http://localhost:7860

> **首次启动说明**: 
> - 首次运行会自动下载 VoxCPM 基础模型（~1.5 GB）
> - 下载时间取决于网络速度，通常需要 5-15 分钟
> - 模型会缓存到本地，后续启动只需 10 秒

详细说明请查看 [INSTALL.md](INSTALL.md)

## 📖 使用说明

打开浏览器访问 http://localhost:7860 即可使用 Web 界面。

详细使用说明请查看 [文档目录](docs/)。



## 🎯 技术细节

- **基础模型**: VoxCPM 1.5 (openbmb/VoxCPM1.5)
- **微调方法**: LoRA (Rank: 32)
- **采样率**: 44100 Hz
- **LoRA 模型**: [ErnarBahat/VoxCPM-KazakhTTS-Lora](https://huggingface.co/ErnarBahat/VoxCPM-KazakhTTS-Lora)

## ⚠️ 常见问题

- **首次启动很慢**: 首次运行需要下载基础模型（~1.5 GB），需要 5-15 分钟。后续启动只需 10 秒。
- **端口被占用**: 修改 `web_app.py` 中的端口号，或停止占用 7860 端口的进程。
- **CUDA 不可用**: 系统会自动使用 CPU 模式（速度较慢但可用）。

更多问题请查看 [INSTALL.md](INSTALL.md)。

## 📊 系统要求

**最低配置**: Python 3.8+, 8GB RAM, 4GB 磁盘

**推荐配置**: Python 3.10+, 16GB RAM, NVIDIA GPU (6GB+ VRAM), 10GB 磁盘



## 📄 许可证

本项目基于 VoxCPM 1.5 开发，遵循相应的开源许可证。

## 🙏 致谢

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) - 基础 TTS 模型
- [Gradio](https://gradio.app/) - Web 界面框架
- [Hugging Face](https://huggingface.co/) - 模型托管

## 📞 支持

如有问题或建议，请查看项目文档：
- [INSTALL.md](INSTALL.md) - 安装指南
- [docs/](docs/) - 用户指南

---

**享受多语言语音合成！** 🎉
