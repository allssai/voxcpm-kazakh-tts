# VoxCPM Kazakh TTS - Multilingual Text-to-Speech Engine

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)

A multilingual text-to-speech system based on VoxCPM 1.5, with LoRA fine-tuning specifically optimized for Kazakh language.

[Quick Start](#-quick-start) • [Documentation](INSTALL.md) • [中文文档](README_CN.md)

</div>

---

## ✨ Key Features

- 🌍 **Multilingual Support**: Kazakh, Chinese, English, and mixed text
- 🎭 **Zero-Shot Voice Cloning**: Clone any voice with 3-10 seconds of reference audio
- ⚡ **Real-Time Synthesis**: Fast high-quality speech generation
- 🎯 **Kazakh Optimization**: Enhanced Kazakh language quality through LoRA fine-tuning
- 🎨 **Bilingual Interface**: Full support for Chinese and Kazakh
- 📊 **Real-Time Progress**: Display inference process and detailed logs
- 🔧 **Flexible Configuration**: Adjustable speed, pitch, inference steps, and more
- 🎵 **Voice Management**: Complete CRUD operations for voice presets

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- 8GB+ RAM
- 4GB+ Disk Space
- (Recommended) NVIDIA GPU with CUDA

### Three-Step Deployment

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/voxcpm-kazakh-tts.git
cd voxcpm-kazakh-tts
```

#### 2. Download LoRA Model

The LoRA model is hosted on HuggingFace and needs to be downloaded separately:

```bash
# Method 1: Using huggingface-cli (Recommended)
pip install huggingface_hub
huggingface-cli download ErnarBahat/VoxCPM-KazakhTTS-Lora --local-dir ./lora

# Method 2: Manual Download
# Visit https://huggingface.co/ErnarBahat/VoxCPM-KazakhTTS-Lora
# Download all files to ./lora/ directory
```

#### 3. Install and Launch

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh    # Auto-install dependencies
./start.sh      # Launch application
```

**Windows:**
```bash
install.bat     # Auto-install dependencies
start.bat       # Launch application
```

**Docker:**
```bash
docker-compose up -d
```

#### 4. Access the Application

Open your browser and visit: http://localhost:7860

> **First Launch Note**: 
> - The first run will automatically download the VoxCPM base model (~1.5 GB)
> - Download time depends on network speed, typically 5-15 minutes
> - The model will be cached locally, subsequent launches take only 10 seconds

For detailed instructions, see [INSTALL.md](INSTALL.md)

## 📖 Usage

Open your browser and visit http://localhost:7860 to access the web interface.

For detailed usage instructions, see the [documentation](docs/).

## 🎯 Technical Details

- **Base Model**: VoxCPM 1.5 (openbmb/VoxCPM1.5)
- **Fine-tuning**: LoRA (Rank: 32)
- **Sample Rate**: 44100 Hz
- **LoRA Model**: [ErnarBahat/VoxCPM-KazakhTTS-Lora](https://huggingface.co/ErnarBahat/VoxCPM-KazakhTTS-Lora)

## ⚠️ Common Issues

- **First launch is slow**: The first run downloads the base model (~1.5 GB), taking 5-15 minutes. Subsequent launches take only 10 seconds.
- **Port already in use**: Change the port in `web_app.py` or stop the process using port 7860.
- **CUDA not available**: The system will automatically use CPU mode (slower but functional).

For more troubleshooting, see [INSTALL.md](INSTALL.md).

## 📊 System Requirements

**Minimum**: Python 3.8+, 8GB RAM, 4GB Disk

**Recommended**: Python 3.10+, 16GB RAM, NVIDIA GPU (6GB+ VRAM), 10GB Disk



## 📄 License

This project is based on VoxCPM 1.5 and follows the corresponding open-source license.

## 🙏 Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) - Base TTS model
- [Gradio](https://gradio.app/) - Web interface framework
- [Hugging Face](https://huggingface.co/) - Model hosting

## 📞 Support

For questions or suggestions, please check the project documentation:
- [INSTALL.md](INSTALL.md) - Installation guide
- [docs/](docs/) - User guides

---

**Enjoy multilingual speech synthesis!** 🎉
