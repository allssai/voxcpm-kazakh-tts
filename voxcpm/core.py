import os
import sys
import re
import tempfile
import numpy as np
from typing import Generator, Optional
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel, LoRAConfig

class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            enable_denoiser : bool = True,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
        ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: ModelScope acoustic noise suppression model
                id or local path. If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is 
                provided without lora_config, a default config will be created.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded.
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}", file=sys.stderr)
        
        # If lora_weights_path is provided but no lora_config, create a default one
        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )
            print(f"Auto-created default LoRAConfig for loading weights from: {lora_weights_path}", file=sys.stderr)
        
        self.tts_model = VoxCPMModel.from_local(voxcpm_model_path, optimize=optimize, lora_config=lora_config)
        
        # Load LoRA weights if path is provided
        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}", file=sys.stderr)
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            print(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}", file=sys.stderr)
        
        self.text_normalizer = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from .zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        if optimize:
            print("Warm up VoxCPMModel...", file=sys.stderr)
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM1.5",
            load_denoiser: bool = True,
            zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            cache_dir: str = None,
            local_files_only: bool = False,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
            **kwargs,
        ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            load_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is 
                provided without lora_config, a default config will be created with
                enable_lm=True and enable_dit=True.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded
                after model initialization.
        Kwargs:
            Additional keyword arguments passed to the ``VoxCPM`` constructor.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")
        
        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            min_len : int = 2,
            max_len : int = 4096,
            normalize : bool = False,
            denoise : bool = False,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        This method optionally builds and reuses a prompt cache. If an external
        prompt (``prompt_wav_path`` + ``prompt_text``) is provided, it will be
        used for all sub-sentences. Otherwise, the prompt cache is built from
        the first generated result and reused for the remaining text chunks.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_len: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt audio if a denoiser is
                available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU. 
            Yields audio chunks for each generations step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")
        
        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")
        
        # Allow prompt_wav_path without prompt_text for direct audio prompting
        if prompt_wav_path is not None and prompt_text is None:
            prompt_text = ""
        
        # 导入 re 模块（在函数开始处导入，避免作用域问题）
        import re
        
        # 修复：不应该全局替换换行，而是利用它进行分句
        # text = text.replace("\n", " ")
        # text = re.sub(r'\s+', ' ', text)
        temp_prompt_wav_path = None
        
        try:
            # 如果提供了参考音频，构建 prompt cache
            # 关键修复：prompt_text 必须是参考音频的实际文本内容，不能为空
            # 如果用户没有提供，应该跳过音色克隆而不是传空字符串
            if prompt_wav_path is not None:
                if denoise and self.denoiser is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_prompt_wav_path = tmp_file.name
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_prompt_wav_path)
                    prompt_wav_path = temp_prompt_wav_path
                
                # 关键修复：处理空参考文本的情况
                # 即使参考文本为空，也要使用音频特征（仅音频模式）
                if prompt_text and prompt_text.strip():
                    # 有参考文本：正常处理
                    def detect_language_type(text):
                        """检测文本的主要语言类型"""
                        # 统计不同字符类型的数量
                        latin_count = len(re.findall(r'[a-zA-Z]', text))
                        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ]', text))
                        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
                        
                        total_chars = latin_count + cyrillic_count + chinese_count
                        if total_chars == 0:
                            return "unknown"
                        
                        # 计算各语言占比
                        latin_ratio = latin_count / total_chars
                        cyrillic_ratio = cyrillic_count / total_chars
                        chinese_ratio = chinese_count / total_chars
                        
                        # 判断主要语言
                        # 优先级：西里尔 > 中文 > 拉丁（因为哈萨克语可能包含英文术语）
                        if cyrillic_ratio > 0.2:  # 降低阈值，只要有20%西里尔字母就认为是哈萨克语
                            return "cyrillic"  # 哈萨克语/俄语为主
                        elif chinese_ratio > 0.2:  # 只要有20%中文就认为是中文
                            return "chinese"  # 中文为主
                        elif latin_ratio > 0.6:  # 拉丁字母需要超过60%才认为是纯英文
                            return "latin"  # 英文为主
                        else:
                            return "mixed"  # 混合语言
                    
                    prompt_lang = detect_language_type(prompt_text)
                    target_lang = detect_language_type(text)
                    
                    # 判断是否为跨语言场景
                    is_cross_language = (prompt_lang != target_lang)
                    
                    if is_cross_language:
                        print(f">>> 检测到跨语言音色克隆（参考: {prompt_lang}, 目标: {target_lang}）", file=sys.stderr)
                        print(f">>> 使用分隔模式：保留参考文本提供音色信息，用句号分隔避免内容泄露", file=sys.stderr)
                    else:
                        print(f">>> 同语言音色克隆（{prompt_lang}），使用标准模式", file=sys.stderr)
                    
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text.strip(),
                        use_prompt_text=True,  # 始终使用参考文本以保证音色质量
                        cross_language=is_cross_language  # 标记是否跨语言
                    )
                    print(f">>> 使用音色克隆，参考文本: {prompt_text.strip()[:50]}...", file=sys.stderr)
                else:
                    # 参考文本为空：使用仅音频模式
                    print(f">>> 参考文本为空，使用仅音频模式", file=sys.stderr)
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_wav_path=prompt_wav_path,
                        prompt_text="",  # 空文本
                        use_prompt_text=False,  # 不使用参考文本
                        cross_language=False
                    )
                    print(f">>> 使用音色克隆（仅音频特征）", file=sys.stderr)
            else:
                fixed_prompt_cache = None  # will be built from the first inference
            
            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer
                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)
            
            # 预处理：将换行符替换为空格，避免被当作文本读出
            text = text.replace('\n', ' ').replace('\r', ' ')
            # 清理多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 改进的分句逻辑：更保守的策略，减少卡顿
            # 检测文本主要语言
            def detect_text_language(text):
                """检测文本的主要语言"""
                latin_count = len(re.findall(r'[a-zA-Z]', text))
                cyrillic_count = len(re.findall(r'[а-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ]', text))
                chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
                
                total = latin_count + cyrillic_count + chinese_count
                if total == 0:
                    return "unknown"
                
                if latin_count / total > 0.6:
                    return "latin"  # 英文为主
                elif cyrillic_count / total > 0.3:
                    return "cyrillic"  # 哈萨克语/俄语为主
                elif chinese_count / total > 0.3:
                    return "chinese"  # 中文为主
                else:
                    return "mixed"  # 混合语言
            
            text_lang = detect_text_language(text)
            
            # 根据语言选择分句策略
            if text_lang == "latin":
                # 英文：非常保守的分句策略
                # 只在段落很长（>300字符）时才考虑分句
                if len(text) <= 300:
                    # 短文本：不分句，整段处理
                    raw_sentences = [text]
                else:
                    # 长文本：只在非常明确的句子结束处分句
                    # 匹配：句号/问号/感叹号 + 空格 + 大写字母
                    parts = re.split(r'([.!?]+\s+)(?=[A-Z])', text)
                    raw_sentences = []
                    current = ""
                    for part in parts:
                        current += part
                        # 只有当累积长度超过200字符时才分句
                        if len(current) > 200 and re.search(r'[.!?]+\s+$', current):
                            raw_sentences.append(current.strip())
                            current = ""
                    if current.strip():
                        raw_sentences.append(current.strip())
                    
                    # 如果没有分句成功，整段处理
                    if not raw_sentences:
                        raw_sentences = [text]
            else:
                # 中文/哈萨克语：按句末标点拆分
                parts = re.split(r'([。！？….!?]+)', text)
                raw_sentences = []
                for i in range(0, len(parts) - 1, 2):
                    s = parts[i] + parts[i+1]
                    if s.strip():
                        raw_sentences.append(s)
                if len(parts) % 2 == 1:
                    s = parts[-1]
                    if s.strip():
                        raw_sentences.append(s)
                        
                if not raw_sentences:
                    raw_sentences = [text]

            # 对超长句子（>250字符）进行二次拆分
            sentences = []
            for sent in raw_sentences:
                if len(sent) <= 250:
                    sentences.append(sent)
                else:
                    # 英文：按逗号+空格拆分
                    if text_lang == "latin":
                        sub_parts = re.split(r'(,\s+)', sent)
                    else:
                        # 中文/哈萨克语：按逗号拆分
                        sub_parts = re.split(r'([,，;；]+)', sent)
                    
                    current_chunk = ""
                    for j in range(0, len(sub_parts)):
                        if len(current_chunk) + len(sub_parts[j]) <= 250:
                            current_chunk += sub_parts[j]
                        else:
                            if current_chunk.strip():
                                sentences.append(current_chunk.strip())
                            current_chunk = sub_parts[j]
                    if current_chunk.strip():
                        sentences.append(current_chunk.strip())

            current_prompt_cache = fixed_prompt_cache
            
            print(f">>> 分句结果：共 {len(sentences)} 个句子", file=sys.stderr)
            print(f">>> fixed_prompt_cache: {'存在' if fixed_prompt_cache else '不存在'}", file=sys.stderr)
            
            for i, utt in enumerate(sentences):
                # 根据是否有 prompt 调整 max_len
                # 平衡质量和速度：适度的 max_len
                if current_prompt_cache is not None:
                    # 有 prompt：每个字符约 3-4 个patch
                    # 公式：4倍字符数 + 25，上限 200
                    utt_max_len = min(len(utt) * 4 + 25, 200)
                else:
                    # 无 prompt：每个字符约 2-3 个patch
                    # 公式：3倍字符数 + 25，上限 150
                    utt_max_len = min(len(utt) * 3 + 25, 150)
                
                # 调试信息
                prompt_info = "无" if current_prompt_cache is None else f"有(文本长度={len(current_prompt_cache.get('prompt_text', ''))})"
                print(f"  [{i+1}/{len(sentences)}] 长度={len(utt)}, max_len={utt_max_len}, prompt={prompt_info}, 内容: {utt[:50]}...", file=sys.stderr)
                
                generate_result = self.tts_model._generate_with_prompt_cache(
                                target_text=utt,
                                prompt_cache=current_prompt_cache,
                                min_len=min_len,
                                max_len=utt_max_len,
                                inference_timesteps=inference_timesteps,
                                cfg_value=cfg_value,
                                retry_badcase=False,  # 禁用重试以加速
                                retry_badcase_max_times=1,
                                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                                streaming=streaming,
                            )
            
                last_feat = None
                for wav, _, feat in generate_result:
                    last_feat = feat
                    yield wav.squeeze(0).cpu().numpy()
                
                # 关键修复：使用音频克隆时，始终使用固定的 prompt_cache，不要累积
                # 只有在没有提供外部 prompt 时，才使用前一句的输出作为下一句的 prompt
                if fixed_prompt_cache is None and last_feat is not None and not streaming:
                    current_prompt_cache = {
                        "prompt_text": utt,
                        "audio_feat": last_feat
                    }
        
        finally:
            if temp_prompt_wav_path and os.path.exists(temp_prompt_wav_path):
                try:
                    os.unlink(temp_prompt_wav_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # LoRA Interface (delegated to VoxCPMModel)
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        """Load LoRA weights from a checkpoint file.
        
        Args:
            lora_weights_path: Path to LoRA weights (.pth file or directory
                containing lora_weights.ckpt).
        
        Returns:
            tuple: (loaded_keys, skipped_keys) - lists of loaded and skipped parameter names.
        
        Raises:
            RuntimeError: If model was not initialized with LoRA config.
        """
        if self.tts_model.lora_config is None:
            raise RuntimeError(
                "Cannot load LoRA weights: model was not initialized with LoRA config. "
                "Please reinitialize with lora_config or lora_weights_path parameter."
            )
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        """Unload LoRA by resetting all LoRA weights to initial state (effectively disabling LoRA)."""
        self.tts_model.reset_lora_weights()
    
    def set_lora_enabled(self, enabled: bool):
        """Enable or disable LoRA layers without unloading weights.
        
        Args:
            enabled: If True, LoRA layers are active; if False, only base model is used.
        """
        self.tts_model.set_lora_enabled(enabled)
    
    def get_lora_state_dict(self) -> dict:
        """Get current LoRA parameters state dict.
        
        Returns:
            dict: State dict containing all LoRA parameters (lora_A, lora_B).
        """
        return self.tts_model.get_lora_state_dict()
    
    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA is currently configured."""
        return self.tts_model.lora_config is not None
