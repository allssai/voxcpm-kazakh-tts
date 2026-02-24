import os
import sys
import json
import shutil
import gradio as gr
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# ═══════════════════════════════════════════════════
# 1. 路径与环境设置
# ═══════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(PROJECT_ROOT, "lora")
CONFIG_FILE = os.path.join(LORA_PATH, "lora_config.json")
VOICES_DIR = os.path.join(PROJECT_ROOT, "voices")
MODEL_ID = "openbmb/VoxCPM1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(VOICES_DIR, exist_ok=True)

print(f"--- 启动三语 TTS 界面 ---")
print(f"设备: {DEVICE}")
print(f"模型: {MODEL_ID}")
print(f"音色库: {VOICES_DIR}")

# ═══════════════════════════════════════════════════
# 2. 读取 LoRA 配置并加载模型
# ═══════════════════════════════════════════════════
lora_rank = 32
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
            lora_rank = config_data.get("lora_config", {}).get("r", 32)
            print(f"LoRA Rank: {lora_rank}")
    except Exception as e:
        print(f"读取配置失败: {e}")

try:
    lora_cfg = LoRAConfig(r=lora_rank, enable_lm=True, enable_dit=True, enable_proj=False)
    # 暂时禁用编译优化以确保在 Windows 上的绝对稳定性
    model = VoxCPM.from_pretrained(
        hf_model_id=MODEL_ID, 
        load_denoiser=False, 
        lora_weights_path=LORA_PATH,
        lora_config=lora_cfg,
        optimize=False  # 禁用编译以避免之前出现的 AssertionError
    )
    
    actual_r = model.tts_model.lora_config.r
    print(f"[OK] 模型加载成功！实际 LoRA Rank: {actual_r}")
except Exception as e:
    print(f"[ERROR] 加载失败: {e}, 尝试无 LoRA 启动...")
    model = VoxCPM.from_pretrained(hf_model_id=MODEL_ID, load_denoiser=False, optimize=False)

# ═══════════════════════════════════════════════════
# 3. 多语言文本定义
# ═══════════════════════════════════════════════════
TRANSLATIONS = {
    "zh": {
        # 主界面
        "title": "多语言 TTS 引擎",
        "subtitle": "支持哈萨克语、中文、英文及多语言混合 · 零样本音色克隆 · 实时语音合成",
        "tab_synthesis": "语音合成",
        "tab_voice_management": "音色管理",
        "switch_language": "切换到哈萨克语",
        
        # 语音合成
        "input_label": "待合成内容",
        "input_placeholder": "支持哈萨克语、中文、英文混合...",
        "input_default": "Сәлем! Бұл - халықаралық деңгейдегі дауыс синтезі. 欢迎使用三语语音合成系统。",
        "voice_preset": "音色预设",
        "create_voice": "创建新音色",
        "create_voice_warning": "⚠️ **重要**: 必须填写参考文本（音频中说的内容），否则音色克隆无法正常工作",
        "voice_name": "音色名称",
        "voice_name_placeholder": "例如：专业男声",
        "ref_audio": "参考音频 (3-10秒)",
        "ref_text": "参考文本 (必填)",
        "ref_text_placeholder": "请输入参考音频中说的内容",
        "save_voice": "保存音色",
        "delete_voice": "删除选中音色",
        "operation_status": "操作状态",
        "enable_lora": "启用哈萨克语 LoRA (推荐)",
        "advanced_params": "高级参数",
        "inference_steps": "推理步数",
        "cfg_strength": "CFG 强度",
        "speed": "语速",
        "pitch": "音调",
        "remove_silence": "移除过长静音 (>0.8秒)",
        "temp_voice_clone": "临时音色克隆",
        "temp_voice_warning": "仅本次有效，不保存。**必须填写参考文本**。",
        "generate_btn": "开始合成",
        "audio_output": "音频输出",
        "status_info": "状态信息",
        "inference_log": "推理日志",
        "log_placeholder": "推理过程将在这里显示...",
        "usage_tips": """### 使用提示
- **哈萨克语合成**: 请启用 LoRA 以获得最佳效果
- **音色克隆**: 必须填写参考文本（音频中说的内容）
- **多语言混合**: 支持哈萨克语、中文、英文自由混合
- **停顿控制**: 使用标点符号（。！？，...）控制停顿""",
        
        # 音色管理
        "voice_mgmt_title": "音色管理中心",
        "voice_mgmt_intro": """完整的音色增删改查功能，支持：
- ✅ 查看所有音色状态
- ➕ 创建新音色
- ✏️ 编辑参考文本和音频
- 🗑️ 删除音色
- 📊 实时对齐验证

**重要**: 参考文本必须与音频内容100%匹配！""",
        
        # 音色列表
        "tab_voice_list": "音色列表",
        "voice_list_title": "所有音色状态",
        "refresh_status": "刷新状态",
        
        # 创建音色
        "tab_create_voice": "创建音色",
        "create_voice_title": "创建新音色",
        "create_voice_steps": """**步骤**:
1. 输入音色名称（英文或拼音，不要有空格）
2. 上传参考音频（3-10秒，清晰无噪音）
3. 输入参考文本（音频中说的内容，必须100%准确）
4. 点击创建""",
        "voice_name_info": "只能包含字母、数字、下划线",
        "voice_name_example": "例如: my_voice_1",
        "ref_text_info": "必须与音频内容100%一致",
        "create_btn": "创建音色",
        "create_status": "创建状态",
        
        # 编辑音色
        "tab_edit_voice": "编辑音色",
        "edit_voice_title": "编辑音色",
        "edit_voice_intro": """**编辑参考文本**:
1. 选择音色
2. 播放音频，听清楚内容
3. 修改参考文本
4. 保存

**更新音频文件**:
1. 选择音色
2. 上传新的音频文件
3. 更新后记得检查参考文本是否仍然匹配""",
        "select_voice": "选择音色",
        "please_select_voice": "请选择音色",
        "current_ref_audio": "当前参考音频",
        "update_audio_file": "更新音频文件",
        "new_ref_audio": "新的参考音频",
        "update_audio_btn": "更新音频",
        "ref_text_label": "参考文本",
        "edit_tips": """**提示**:
- 播放音频，仔细听清楚内容
- 输入音频中实际说的内容
- 不能多也不能少
- 包括正确的标点符号""",
        "save_ref_text": "保存参考文本",
        "clear_btn": "清空",
        
        # 查看详情
        "tab_view_details": "查看详情",
        "view_details_title": "音色详细信息",
        
        # 删除音色
        "tab_delete_voice": "删除音色",
        "delete_voice_title": "删除音色",
        "delete_warning": """⚠️ **警告**: 删除操作不可恢复！

删除音色会：
- 删除音频文件
- 删除元数据
- 删除整个音色目录

请谨慎操作！""",
        "select_voice_to_delete": "选择要删除的音色",
        "delete_confirm": "我确认要删除此音色（不可恢复）",
        "delete_btn": "删除音色",
        "delete_status": "删除状态",
        
        # 错误和成功消息
        "error_no_name": "[错误] 请输入音色名称",
        "error_no_audio": "[错误] 请上传参考音频",
        "error_no_ref_text": "[错误] 请输入参考文本（音频中说的内容），这是音色克隆的关键！",
        "error_no_text": "[错误] 请输入文本",
        "error_select_voice": "[错误] 请选择音色",
        "error_voice_exists": "[错误] 音色 '{voice_name}' 已存在，请使用其他名称",
        "error_voice_not_exist": "[错误] 音色 '{voice_name}' 不存在",
        "error_no_new_audio": "[错误] 请上传新的音频文件",
        "error_confirm_delete": "[错误] 请先勾选确认框",
        "success_voice_created": "[成功] 音色 '{voice_name}' 创建成功！",
        "success_voice_deleted": "[成功] 音色 '{voice_name}' 已删除",
        "success_ref_text_saved": "[成功] 已保存 {voice_name} 的参考文本",
        "success_audio_updated": "[成功] 已更新 {voice_name} 的音频文件",
        "warning_check_ref_text": "⚠️ 请检查参考文本是否仍然匹配新音频",
        
        "footer": "多语言 TTS 引擎 - 哈萨克语强化版"
    },
    "kk": {
        # 主界面
        "title": "Көптілді TTS жүйесі",
        "subtitle": "Қазақ, қытай, ағылшын тілдерін қолдайды · Үлгісіз дауыс клондау · Нақты уақыт режимінде сөйлеу синтезі",
        "tab_synthesis": "Сөйлеу синтезі",
        "tab_voice_management": "Дауыс басқару",
        "switch_language": "中文切换",
        
        # 语音合成
        "input_label": "Синтездеу мәтіні",
        "input_placeholder": "Қазақ, қытай, ағылшын тілдерін араластыруға болады...",
        "input_default": "Сәлем! Бұл - халықаралық деңгейдегі дауыс синтезі. 欢迎使用三语语音合成系统。",
        "voice_preset": "Дауыс үлгісі",
        "create_voice": "Жаңа дауыс жасау",
        "create_voice_warning": "⚠️ **Маңызды**: Анықтама мәтінін (аудиодағы мазмұн) міндетті түрде толтырыңыз, әйтпесе дауыс клондау жұмыс істемейді",
        "voice_name": "Дауыс атауы",
        "voice_name_placeholder": "Мысалы: Кәсіби ер адам дауысы",
        "ref_audio": "Анықтама аудио (3-10 секунд)",
        "ref_text": "Анықтама мәтіні (міндетті)",
        "ref_text_placeholder": "Анықтама аудиодағы мазмұнды енгізіңіз",
        "save_voice": "Дауысты сақтау",
        "delete_voice": "Таңдалған дауысты жою",
        "operation_status": "Операция күйі",
        "enable_lora": "Қазақ тілі LoRA қосу (ұсынылады)",
        "advanced_params": "Қосымша параметрлер",
        "inference_steps": "Қорытынды қадамдар",
        "cfg_strength": "CFG күші",
        "speed": "Сөйлеу жылдамдығы",
        "pitch": "Дыбыс биіктігі",
        "remove_silence": "Ұзақ үнсіздікті жою (>0.8с)",
        "temp_voice_clone": "Уақытша дауыс клондау",
        "temp_voice_warning": "Тек осы рет үшін жарамды, сақталмайды. **Анықтама мәтінін міндетті түрде толтырыңыз**.",
        "generate_btn": "Синтездеуді бастау",
        "audio_output": "Аудио нәтижесі",
        "status_info": "Күй ақпараты",
        "inference_log": "Қорытынды журналы",
        "log_placeholder": "Қорытынды процесі осында көрсетіледі...",
        "usage_tips": """### Пайдалану кеңестері
- **Қазақ тілін синтездеу**: Ең жақсы нәтиже үшін LoRA қосыңыз
- **Дауыс клондау**: Анықтама мәтінін (аудиодағы мазмұн) міндетті түрде толтырыңыз
- **Көптілді араластыру**: Қазақ, қытай, ағылшын тілдерін еркін араластыруға болады
- **Үзіліс басқару**: Тыныс белгілерін (。！？，...) пайдаланыңыз""",
        
        # 音色管理
        "voice_mgmt_title": "Дауыс басқару орталығы",
        "voice_mgmt_intro": """Дауыстарды толық басқару функциялары:
- ✅ Барлық дауыс күйлерін көру
- ➕ Жаңа дауыс жасау
- ✏️ Анықтама мәтінін және аудионы өңдеу
- 🗑️ Дауысты жою
- 📊 Нақты уақытта туралау тексеру

**Маңызды**: Анықтама мәтіні аудио мазмұнымен 100% сәйкес болуы керек!""",
        
        # 音色列表
        "tab_voice_list": "Дауыс тізімі",
        "voice_list_title": "Барлық дауыс күйлері",
        "refresh_status": "Күйді жаңарту",
        
        # 创建音色
        "tab_create_voice": "Дауыс жасау",
        "create_voice_title": "Жаңа дауыс жасау",
        "create_voice_steps": """**Қадамдар**:
1. Дауыс атауын енгізіңіз (ағылшын немесе латын әріптері, бос орын болмасын)
2. Анықтама аудиосын жүктеңіз (3-10 секунд, таза дыбыс)
3. Анықтама мәтінін енгізіңіз (аудиодағы мазмұн, 100% дәл)
4. Жасау батырмасын басыңыз""",
        "voice_name_info": "Тек әріптер, сандар, астын сызу белгісі",
        "voice_name_example": "Мысалы: menin_dauysym_1",
        "ref_text_info": "Аудио мазмұнымен 100% сәйкес болуы керек",
        "create_btn": "Дауыс жасау",
        "create_status": "Жасау күйі",
        
        # 编辑音色
        "tab_edit_voice": "Дауысты өңдеу",
        "edit_voice_title": "Дауысты өңдеу",
        "edit_voice_intro": """**Анықтама мәтінін өңдеу**:
1. Дауысты таңдаңыз
2. Аудионы ойнатып, мазмұнды тыңдаңыз
3. Анықтама мәтінін өзгертіңіз
4. Сақтаңыз

**Аудио файлын жаңарту**:
1. Дауысты таңдаңыз
2. Жаңа аудио файлын жүктеңіз
3. Жаңартудан кейін анықтама мәтінінің сәйкестігін тексеріңіз""",
        "select_voice": "Дауысты таңдау",
        "please_select_voice": "Дауысты таңдаңыз",
        "current_ref_audio": "Ағымдағы анықтама аудио",
        "update_audio_file": "Аудио файлын жаңарту",
        "new_ref_audio": "Жаңа анықтама аудио",
        "update_audio_btn": "Аудионы жаңарту",
        "ref_text_label": "Анықтама мәтіні",
        "edit_tips": """**Кеңестер**:
- Аудионы ойнатып, мазмұнды мұқият тыңдаңыз
- Аудиодағы нақты мазмұнды енгізіңіз
- Артық немесе кем болмауы керек
- Дұрыс тыныс белгілерін қосыңыз""",
        "save_ref_text": "Анықтама мәтінін сақтау",
        "clear_btn": "Тазалау",
        
        # 查看详情
        "tab_view_details": "Толық ақпарат",
        "view_details_title": "Дауыс туралы толық ақпарат",
        
        # 删除音色
        "tab_delete_voice": "Дауысты жою",
        "delete_voice_title": "Дауысты жою",
        "delete_warning": """⚠️ **Ескерту**: Жою әрекетін қайтару мүмкін емес!

Дауысты жою:
- Аудио файлын жояды
- Метадеректерді жояды
- Бүкіл дауыс каталогын жояды

Абайлап әрекет етіңіз!""",
        "select_voice_to_delete": "Жою үшін дауысты таңдаңыз",
        "delete_confirm": "Мен бұл дауысты жоюды растаймын (қайтарылмайды)",
        "delete_btn": "Дауысты жою",
        "delete_status": "Жою күйі",
        
        # 错误和成功消息
        "error_no_name": "[Қате] Дауыс атауын енгізіңіз",
        "error_no_audio": "[Қате] Анықтама аудиосын жүктеңіз",
        "error_no_ref_text": "[Қате] Анықтама мәтінін (аудиодағы мазмұн) енгізіңіз, бұл дауыс клондаудың кілті!",
        "error_no_text": "[Қате] Мәтінді енгізіңіз",
        "error_select_voice": "[Қате] Дауысты таңдаңыз",
        "error_voice_exists": "[Қате] '{voice_name}' дауысы бар, басқа атау пайдаланыңыз",
        "error_voice_not_exist": "[Қате] '{voice_name}' дауысы жоқ",
        "error_no_new_audio": "[Қате] Жаңа аудио файлын жүктеңіз",
        "error_confirm_delete": "[Қате] Алдымен растау белгісін қойыңыз",
        "success_voice_created": "[Сәтті] '{voice_name}' дауысы жасалды!",
        "success_voice_deleted": "[Сәтті] '{voice_name}' дауысы жойылды",
        "success_ref_text_saved": "[Сәтті] {voice_name} анықтама мәтіні сақталды",
        "success_audio_updated": "[Сәтті] {voice_name} аудио файлы жаңартылды",
        "warning_check_ref_text": "⚠️ Анықтама мәтінінің жаңа аудиомен сәйкестігін тексеріңіз",
        
        "footer": "Көптілді TTS жүйесі - Қазақ тілі нұсқасы"
    }
}

# ═══════════════════════════════════════════════════
# 4. 音色预设管理
# ═══════════════════════════════════════════════════
def _list_voices():
    """扫描 voices/ 目录，返回可用音色名称列表"""
    voices = []
    if os.path.exists(VOICES_DIR):
        for name in sorted(os.listdir(VOICES_DIR)):
            voice_dir = os.path.join(VOICES_DIR, name)
            wav_path = os.path.join(voice_dir, "ref.wav")
            if os.path.isdir(voice_dir) and os.path.exists(wav_path):
                voices.append(name)
    return voices

def get_voice_alignment_info(voice_name):
    """获取音色的对齐信息"""
    import torchaudio
    
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    wav_path = os.path.join(voice_dir, "ref.wav")
    meta_path = os.path.join(voice_dir, "meta.json")
    
    if not os.path.exists(wav_path):
        return None
    
    # 读取音频信息
    audio, sr = torchaudio.load(wav_path)
    duration = audio.shape[1] / sr
    channels = audio.shape[0]
    
    # 读取元数据
    ref_text = ""
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                ref_text = meta.get("ref_text", "")
        except:
            pass
    
    # 计算对齐指标
    word_count = len(ref_text.split()) if ref_text else 0
    estimated_capacity = duration * 2.5
    ratio = word_count / estimated_capacity if estimated_capacity > 0 else 0
    
    # 判断状态
    if not ref_text:
        status = "❌ 空文本"
        status_color = "red"
        score = 0
    elif ratio > 1.3:
        status = "❌ 文本过长"
        status_color = "red"
        score = 30
    elif ratio < 0.4:
        status = "⚠️ 文本过短"
        status_color = "orange"
        score = 60
    elif ratio > 1.1:
        status = "⚠️ 文本略长"
        status_color = "orange"
        score = 80
    elif ratio < 0.6:
        status = "⚠️ 文本略短"
        status_color = "orange"
        score = 80
    else:
        status = "✅ 对齐良好"
        status_color = "green"
        score = 100
    
    return {
        "voice": voice_name,
        "duration": duration,
        "channels": channels,
        "sr": sr,
        "word_count": word_count,
        "estimated_capacity": estimated_capacity,
        "ratio": ratio,
        "status": status,
        "status_color": status_color,
        "score": score,
        "ref_text": ref_text,
        "wav_path": wav_path
    }

def get_all_voices_status():
    """获取所有音色的状态信息"""
    voices = _list_voices()
    results = []
    
    for voice in voices:
        info = get_voice_alignment_info(voice)
        if info:
            results.append(info)
    
    # 按评分排序（问题音色在前）
    results.sort(key=lambda x: (x['score'], x['voice']))
    
    return results

def format_voices_table(lang="zh"):
    """格式化音色状态表格"""
    t = TRANSLATIONS[lang]
    results = get_all_voices_status()
    
    if not results:
        return t.get("error_no_voices", "未找到音色文件" if lang == "zh" else "Дауыс файлы табылмады")
    
    # 统计
    perfect = len([r for r in results if r['score'] == 100])
    good = len([r for r in results if 80 <= r['score'] < 100])
    warning = len([r for r in results if 50 <= r['score'] < 80])
    bad = len([r for r in results if r['score'] < 50])
    
    # 构建表格（根据语言选择列标题）
    if lang == "zh":
        table = "| 音色 | 时长 | 词数 | 容量 | 比例 | 状态 | 评分 |\n"
        stats_label = "统计"
    else:  # kk
        table = "| Дауыс | Ұзақтығы | Сөз саны | Сыйымдылығы | Қатынасы | Күйі | Баллы |\n"
        stats_label = "Статистика"
    
    table += "|------|------|------|------|------|------|------|\n"
    
    for r in results:
        table += f"| {r['voice']} | {r['duration']:.2f}s | {r['word_count']} | {r['estimated_capacity']:.1f} | {r['ratio']:.2f} | {r['status']} | {r['score']} |\n"
    
    table += f"\n\n**{stats_label}**: ✅ {perfect} | ⚠️ {good} | ⚠️ {warning} | ❌ {bad}"
    
    return table

def load_voice_for_edit(voice_name):
    """加载音色信息用于编辑"""
    if not voice_name:
        return None, "", "", ""
    
    info = get_voice_alignment_info(voice_name)
    if not info:
        return None, "", "", ""
    
    status_text = f"""**音色**: {info['voice']}
**音频**: {info['duration']:.2f}秒, {info['sr']}Hz, {info['channels']}声道
**当前词数**: {info['word_count']} 词
**建议词数**: {info['estimated_capacity']:.1f} 词
**对齐比例**: {info['ratio']:.2f}
**状态**: {info['status']}"""
    
    return info['wav_path'], info['ref_text'], status_text, info['voice']

def save_voice_ref_text(voice_name, new_ref_text, lang="zh"):
    """保存音色的参考文本"""
    t = TRANSLATIONS[lang]
    if not voice_name or not voice_name.strip():
        return t["error_select_voice"], format_voices_table(lang)
    
    if not new_ref_text or not new_ref_text.strip():
        return t["error_no_ref_text"], format_voices_table(lang)
    
    meta_path = os.path.join(VOICES_DIR, voice_name, "meta.json")
    
    try:
        # 读取现有元数据
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        else:
            meta = {"name": voice_name}
        
        old_ref_text = meta.get("ref_text", "")
        
        # 备份旧文本
        if old_ref_text:
            meta["_backup_ref_text"] = old_ref_text
        
        # 更新参考文本
        meta["ref_text"] = new_ref_text.strip()
        meta["_manually_aligned"] = True
        meta["_note"] = "参考文本已手动对齐" if lang == "zh" else "Анықтама мәтіні қолмен туралады"
        
        # 移除旧标记
        meta.pop("_original_ref_text", None)
        
        # 保存
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 验证对齐
        info = get_voice_alignment_info(voice_name)
        
        success_msg = t["success_ref_text_saved"].format(voice_name=voice_name) if "success_ref_text_saved" in t else f"✅ 已保存 {voice_name} 的参考文本"
        return f"{success_msg}\n\n{info['status']} (评分: {info['score']})", format_voices_table(lang)
    
    except Exception as e:
        error_msg = t.get("error_save_failed", f"❌ 保存失败: {e}")
        return error_msg, format_voices_table(lang)

def create_new_voice(voice_name, audio_file, ref_text, lang="zh"):
    """创建新音色"""
    t = TRANSLATIONS[lang]
    if not voice_name or not voice_name.strip():
        return t["error_no_name"], format_voices_table(lang), gr.update()
    
    if audio_file is None:
        return t["error_no_audio"], format_voices_table(lang), gr.update()
    
    if not ref_text or not ref_text.strip():
        return t["error_no_ref_text"], format_voices_table(lang), gr.update()
    
    voice_name = voice_name.strip()
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    
    # 检查是否已存在
    if os.path.exists(voice_dir):
        return t["error_voice_exists"].format(voice_name=voice_name), format_voices_table(lang), gr.update()
    
    try:
        # 创建目录
        os.makedirs(voice_dir, exist_ok=True)
        
        # 复制音频文件
        dst_wav = os.path.join(voice_dir, "ref.wav")
        shutil.copy2(audio_file, dst_wav)
        
        # 保存元数据
        meta = {
            "name": voice_name,
            "ref_text": ref_text.strip(),
            "_created_via": "web_interface"
        }
        with open(os.path.join(voice_dir, "meta.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # 验证对齐
        info = get_voice_alignment_info(voice_name)
        
        new_choices = _list_voices()
        success_msg = t["success_voice_created"].format(voice_name=voice_name)
        return (
            f"{success_msg}\n\n{info['status']} (评分: {info['score']})",
            format_voices_table(lang),
            gr.update(choices=new_choices, value=voice_name)
        )
    
    except Exception as e:
        # 清理失败的创建
        if os.path.exists(voice_dir):
            shutil.rmtree(voice_dir)
        error_msg = t.get("error_create_failed", f"❌ 创建失败: {e}")
        return error_msg, format_voices_table(lang), gr.update()

def delete_voice_from_management(voice_name, lang="zh"):
    """从音色管理删除音色"""
    t = TRANSLATIONS[lang]
    if not voice_name or not voice_name.strip():
        return t["error_select_voice"], format_voices_table(lang), gr.update()
    
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    
    if not os.path.exists(voice_dir):
        return t["error_voice_not_exist"].format(voice_name=voice_name), format_voices_table(lang), gr.update()
    
    try:
        # 删除目录
        shutil.rmtree(voice_dir)
        
        new_choices = _list_voices()
        default_voice = new_choices[0] if new_choices else None
        
        success_msg = t["success_voice_deleted"].format(voice_name=voice_name)
        return (
            success_msg,
            format_voices_table(lang),
            gr.update(choices=new_choices, value=default_voice)
        )
    
    except Exception as e:
        error_msg = t.get("error_delete_failed", f"❌ 删除失败: {e}")
        return error_msg, format_voices_table(lang), gr.update()

def get_voice_details(voice_name):
    """获取音色详细信息用于查看"""
    if not voice_name:
        return "请选择音色", "", None, ""
    
    info = get_voice_alignment_info(voice_name)
    if not info:
        return "音色不存在", "", None, ""
    
    # 读取完整元数据
    meta_path = os.path.join(VOICES_DIR, voice_name, "meta.json")
    meta_info = ""
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                meta_info = json.dumps(meta, ensure_ascii=False, indent=2)
        except:
            meta_info = "无法读取元数据"
    
    details = f"""### 音色详情: {voice_name}

**音频信息**:
- 时长: {info['duration']:.2f} 秒
- 采样率: {info['sr']} Hz
- 声道: {info['channels']}

**参考文本**:
- 词数: {info['word_count']}
- 建议词数: {info['estimated_capacity']:.1f}
- 对齐比例: {info['ratio']:.2f}

**对齐状态**: {info['status']}
**评分**: {info['score']}/100

**元数据**:
```json
{meta_info}
```
"""
    
    return details, info['ref_text'], info['wav_path'], voice_name

def update_voice_audio(voice_name, new_audio_file, lang="zh"):
    """更新音色的音频文件"""
    t = TRANSLATIONS[lang]
    if not voice_name or not voice_name.strip():
        return t["error_select_voice"], format_voices_table(lang)
    
    if new_audio_file is None:
        return t["error_no_new_audio"], format_voices_table(lang)
    
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    
    if not os.path.exists(voice_dir):
        return t["error_voice_not_exist"].format(voice_name=voice_name), format_voices_table(lang)
    
    try:
        # 备份旧音频
        dst_wav = os.path.join(voice_dir, "ref.wav")
        backup_wav = os.path.join(voice_dir, "ref.wav.backup")
        
        if os.path.exists(dst_wav):
            shutil.copy2(dst_wav, backup_wav)
        
        # 复制新音频
        shutil.copy2(new_audio_file, dst_wav)
        
        # 验证对齐
        info = get_voice_alignment_info(voice_name)
        
        success_msg = t["success_audio_updated"].format(voice_name=voice_name)
        warning_msg = t["warning_check_ref_text"]
        return f"{success_msg}\n\n{info['status']} (评分: {info['score']})\n\n{warning_msg}", format_voices_table(lang)
    
    except Exception as e:
        # 恢复备份
        if os.path.exists(backup_wav):
            shutil.copy2(backup_wav, dst_wav)
        error_msg = t.get("error_update_failed", f"❌ 更新失败: {e}")
        return error_msg, format_voices_table(lang)

def _get_voice_path(voice_name):
    """获取指定音色的参考音频路径和文本"""
    if not voice_name:
        return None, None
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    wav_path = os.path.join(voice_dir, "ref.wav")
    meta_path = os.path.join(voice_dir, "meta.json")
    ref_text = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                ref_text = meta.get("ref_text", None)
        except:
            pass
    if os.path.exists(wav_path):
        return wav_path, ref_text
    return None, None

def create_voice(voice_name, audio_file, ref_text_input, lang="zh"):
    """创建新音色预设"""
    t = TRANSLATIONS[lang]
    if not voice_name or not voice_name.strip():
        return gr.update(), t["error_no_name"]
    if audio_file is None:
        return gr.update(), t["error_no_audio"]
    if not ref_text_input or not ref_text_input.strip():
        return gr.update(), t["error_no_ref_text"]

    voice_name = voice_name.strip()
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    os.makedirs(voice_dir, exist_ok=True)

    # 复制音频文件
    dst_wav = os.path.join(voice_dir, "ref.wav")
    shutil.copy2(audio_file, dst_wav)

    # 保存元数据
    meta = {
        "name": voice_name,
        "ref_text": ref_text_input.strip(),
    }
    with open(os.path.join(voice_dir, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] 音色 '{voice_name}' 已保存到 {voice_dir}")
    new_choices = _list_voices()
    return gr.update(choices=new_choices, value=voice_name), t["success_voice_created"].format(voice_name=voice_name)

def delete_voice(voice_name, lang="zh"):
    """删除音色预设"""
    t = TRANSLATIONS[lang]
    if not voice_name:
        return gr.update(), t["warning_select_voice"]
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    if os.path.exists(voice_dir):
        shutil.rmtree(voice_dir)
        print(f"[DELETE] 音色 '{voice_name}' 已删除")
    new_choices = _list_voices()
    default_voice = "kazakh_man_1" if "kazakh_man_1" in new_choices else (new_choices[0] if new_choices else None)
    return gr.update(choices=new_choices, value=default_voice), t["success_voice_deleted"].format(voice_name=voice_name)

# ═══════════════════════════════════════════════════
# 5. 音频后处理
# ═══════════════════════════════════════════════════
def remove_silence(audio, sr, top_db=40, frame_length=2048, hop_length=512):
    """
    移除音频中过长的静音段（仅移除超过0.8秒的静音）
    
    参数:
        audio: 音频数组
        sr: 采样率
        top_db: 静音阈值（分贝），低于此值视为静音
        frame_length: 帧长度
        hop_length: 跳跃长度
    """
    try:
        # 使用更保守的阈值（40dB）避免误判正常语音为静音
        intervals = librosa.effects.split(
            audio, 
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        if len(intervals) == 0:
            return audio
        
        # 只移除过长的静音段（>0.8秒），保留正常的语音停顿
        segments = []
        max_silence_duration = 0.8  # 最大允许的静音时长（秒）
        max_silence_samples = int(max_silence_duration * sr)
        
        for i, (start, end) in enumerate(intervals):
            # 添加当前语音段
            segments.append(audio[start:end])
            
            # 检查到下一段之间的静音长度
            if i < len(intervals) - 1:
                next_start = intervals[i + 1][0]
                silence_length = next_start - end
                
                if silence_length > max_silence_samples:
                    # 静音过长，缩短到0.5秒
                    silence_samples = int(0.5 * sr)
                    silence_padding = np.zeros(silence_samples, dtype=audio.dtype)
                    segments.append(silence_padding)
                    print(f"  [SILENCE] 缩短静音: {silence_length/sr:.2f}秒 -> 0.5秒")
                else:
                    # 静音正常，保留原始静音
                    segments.append(audio[end:next_start])
        
        result = np.concatenate(segments) if segments else audio
        
        original_duration = len(audio) / sr
        new_duration = len(result) / sr
        reduction = original_duration - new_duration
        
        if reduction > 0.1:
            print(f"  [SILENCE] 原始时长: {original_duration:.2f}秒, 处理后: {new_duration:.2f}秒, 移除: {reduction:.2f}秒")
        else:
            print(f"  [SILENCE] 未检测到过长静音，保持原始音频")
        
        return result
    except Exception as e:
        print(f"  [WARNING] 静音移除失败: {e}, 返回原始音频")
        return audio

def apply_speech_control(y, sr, speed=1.0, pitch=0):
    if speed == 1.0 and pitch == 0:
        return y
    y = y.astype(np.float32)
    if pitch != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
    if speed != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed)
    return y

# ═══════════════════════════════════════════════════
# 6. TTS 合成核心逻辑
# ═══════════════════════════════════════════════════
def cleanup_old_audio_files():
    """清理旧的临时音频文件，保留最近的 10 个"""
    try:
        audio_files = [f for f in os.listdir('.') if f.startswith('output_audio_') and f.endswith('.wav')]
        if len(audio_files) > 10:
            # 按修改时间排序，删除最旧的
            audio_files.sort(key=lambda x: os.path.getmtime(x))
            for old_file in audio_files[:-10]:
                try:
                    os.remove(old_file)
                    print(f"[CLEANUP] 已删除旧文件: {old_file}")
                except:
                    pass
    except Exception as e:
        print(f"[WARNING] 清理旧文件失败: {e}")

def tts_generate(text, voice_preset, ref_audio_upload, ref_text_upload,
                 use_lora, timesteps, cfg, speed_rate, pitch_shift, remove_silence_enabled, lang="zh", progress=gr.Progress()):
    """
    TTS 合成核心逻辑：严格对齐 11 个输入参数 + 进度显示
    """
    t = TRANSLATIONS[lang]
    if not text.strip():
        return None, t["error_no_text"], ""

    progress(0, desc="🔧 初始化...")
    
    status_log = []
    status_log.append(f"📝 收到合成请求: {text[:50]}...")
    status_log.append(f"⚙️ 配置: 音色={voice_preset}, LoRA={use_lora}, 推理步数={timesteps}")
    
    print(f">>> 收到合成请求: {text[:30]}...")
    print(f"    音色: {voice_preset}, LoRA: {use_lora}, Steps: {timesteps}")
    
    # 清理旧文件
    cleanup_old_audio_files()

    try:
        progress(0.1, desc="🎯 配置模型...")
        model.set_lora_enabled(use_lora)
        status_log.append(f"✓ LoRA 状态: {'已启用' if use_lora else '已禁用'}")

        prompt_wav = None
        prompt_txt = None

        # 优先使用上传的参考音频
        if ref_audio_upload is not None:
            prompt_wav = ref_audio_upload
            prompt_txt = ref_text_upload.strip() if (ref_text_upload and ref_text_upload.strip()) else None
            if prompt_txt:
                status_log.append(f"🎤 音色来源: 上传音频")
                status_log.append(f"📄 参考文本: {prompt_txt[:30]}...")
                print(f"    来源: 上传音频 (参考文本: {prompt_txt[:30]}...)")
            else:
                status_log.append(f"⚠️ 上传音频但无参考文本，将使用默认音色")
                print(f"    来源: 上传音频 (无参考文本，将使用默认音色)")
        # 其次使用选定的音色预设
        elif voice_preset:
            wav_path, ref_text = _get_voice_path(voice_preset)
            if wav_path:
                prompt_wav = wav_path
                prompt_txt = ref_text if (ref_text and ref_text.strip()) else None
                if prompt_txt:
                    status_log.append(f"🎭 音色来源: 预设 '{voice_preset}'")
                    status_log.append(f"📄 参考文本: {prompt_txt[:30]}...")
                    print(f"    来源: 音色预设 '{voice_preset}' (参考文本: {prompt_txt[:30]}...)")
                else:
                    status_log.append(f"⚠️ 音色预设 '{voice_preset}' 无参考文本，将使用模型默认音色")
                    print(f"    来源: 音色预设 '{voice_preset}' (无参考文本，将使用模型默认音色)")
            else:
                status_log.append(f"⚠️ 音色预设 '{voice_preset}' 不存在，将使用模型默认音色")
        else:
            status_log.append(f"🎵 使用模型默认音色")

        progress(0.2, desc="🚀 开始推理...")
        
        import time
        start_time = time.time()
        
        status_log.append(f"🔄 正在生成音频...")
        status_log.append(f"⏳ 模型推理中，首次生成可能需要较长时间（通常5-30秒），请耐心等待...")
        status_log.append(f"⏰ 开始时间: {time.strftime('%H:%M:%S')}")
        
        print(f">>> 开始模型推理... 时间: {time.strftime('%H:%M:%S')}", file=sys.stderr)
        print(f">>> 注意：模型推理过程可能需要一些时间，特别是首次生成", file=sys.stderr)
        
        generator = model._generate(
            text=text,
            prompt_wav_path=prompt_wav,
            prompt_text=prompt_txt,
            inference_timesteps=int(timesteps),
            cfg_value=float(cfg),
            streaming=False
        )

        sr = 44100
        if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'sample_rate'):
            sr = model.tts_model.sample_rate

        print(f">>> 开始收集音频chunks...", file=sys.stderr)
        full_audio = []
        chunk_count = 0
        
        progress(0.3, desc="🎵 生成音频中...")
        status_log.append(f"⏳ 模型正在生成音频，请稍候...")
        
        try:
            for chunk in generator:
                chunk_count += 1
                chunk_samples = len(chunk)
                chunk_duration = chunk_samples/sr
                
                # 更新进度（30% - 80%）
                progress_val = 0.3 + (chunk_count * 0.1)
                if progress_val > 0.8:
                    progress_val = 0.8
                progress(progress_val, desc=f"🎵 生成第 {chunk_count} 段音频...")
                
                status_log.append(f"  ✓ 第 {chunk_count} 段: {chunk_samples} 采样点 ({chunk_duration:.2f}秒)")
                print(f"  [CHUNK {chunk_count}] 收到音频: {chunk_samples} samples ({chunk_duration:.2f}秒)", file=sys.stderr)
                full_audio.append(chunk)
        except Exception as chunk_error:
            print(f"  [ERROR] 收集chunk时出错: {chunk_error}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            status_log.append(f"❌ 错误: {str(chunk_error)}")
        
        status_log.append(f"📊 共生成 {chunk_count} 段音频")
        elapsed_time = time.time() - start_time
        status_log.append(f"⏱️ 生成耗时: {elapsed_time:.1f}秒")
        print(f"  [TOTAL] 共收到 {chunk_count} 个chunks, 耗时: {elapsed_time:.1f}秒", file=sys.stderr)
        
        if len(full_audio) == 0:
            return None, "[ERROR] 模型未生成任何音频，请检查输入文本或参数设置", "\n".join(status_log)
            
        progress(0.85, desc="🔧 处理音频...")
        
        final_wav = np.concatenate(full_audio)
        
        # 检查音频是否为空
        if len(final_wav) == 0:
            status_log.append("❌ 错误: 生成的音频长度为零")
            return None, "[ERROR] 生成的音频长度为零，请尝试增加推理步数或调整参数", "\n".join(status_log)
        
        max_amplitude = np.max(np.abs(final_wav))
        if max_amplitude < 1e-6:
            status_log.append("❌ 错误: 生成的音频为静音")
            return None, "[ERROR] 生成的音频为静音，请检查模型状态或更换参考音频", "\n".join(status_log)

        # 移除过长的静音段
        if remove_silence_enabled:
            original_length = len(final_wav)
            final_wav = remove_silence(final_wav, sr, top_db=30)
            removed_samples = original_length - len(final_wav)
            removed_duration = removed_samples / sr
            
            if removed_duration > 0.5:
                status_log.append(f"✂️ 移除静音: {removed_duration:.2f}秒")
                print(f"  [SILENCE] 移除了 {removed_duration:.2f}秒的静音段")
        else:
            status_log.append(f"ℹ️ 静音移除已禁用")

        status_log.append(f"🔧 应用音频控制: 语速={speed_rate}x, 音调={pitch_shift}")
        
        if speed_rate != 1.0 or pitch_shift != 0:
            progress(0.9, desc="🎚️ 调整语速和音调...")
            final_wav = apply_speech_control(final_wav, sr, speed=speed_rate, pitch=pitch_shift)

        progress(0.95, desc="💾 保存音频...")
        
        # 直接转换为 int16，不需要归一化
        # 模型返回的音频已经在正确的振幅范围内
        final_wav = final_wav.astype(np.float32)
        mean_amplitude = np.mean(np.abs(final_wav))
        max_amplitude = np.max(np.abs(final_wav))
        
        status_log.append(f"📊 音频分析: 最大振幅={max_amplitude:.6f}, 平均振幅={mean_amplitude:.6f}")
        print(f"  [DEBUG] 原始音频 max: {max_amplitude:.6f}, mean: {mean_amplitude:.6f}", file=sys.stderr)
        
        # 检测振幅过小的情况（通常发生在使用音色克隆时）
        # 如果平均振幅小于0.001，自动放大50倍
        if mean_amplitude < 0.001 and mean_amplitude > 0:
            amplification_factor = 50
            final_wav = final_wav * amplification_factor
            status_log.append(f"⚡ 检测到振幅过小，自动放大 {amplification_factor}倍")
            print(f"  [WARNING] 检测到振幅过小，自动放大 {amplification_factor}倍", file=sys.stderr)
            print(f"  [DEBUG] 放大后 max: {np.max(np.abs(final_wav)):.6f}, mean: {np.mean(np.abs(final_wav)):.6f}", file=sys.stderr)
        
        # 直接转换为 int16
        final_wav_int16 = (final_wav * 32767).astype(np.int16)
        print(f"  [DEBUG] int16转换后 max: {np.max(np.abs(final_wav_int16))}, mean: {np.mean(np.abs(final_wav_int16)):.2f}", file=sys.stderr)
        
        # 使用带时间戳的唯一文件名，避免 Gradio 和浏览器缓存问题
        import time
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        output_filename = f"output_audio_{timestamp}.wav"
        output_path = os.path.abspath(output_filename)
        wavfile.write(output_path, sr, final_wav_int16)
        
        duration = len(final_wav_int16)/sr
        status_log.append(f"✅ 合成完成！")
        status_log.append(f"📁 文件: {output_filename}")
        status_log.append(f"⏱️ 时长: {duration:.2f}秒")
        status_log.append(f"🎵 采样率: {sr} Hz")
        
        print(f"[OK] 合成成功，样本数: {len(final_wav_int16)}, 路径: {output_path}")
        
        # 验证文件确实写入成功
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            status_log.append("❌ 错误: 音频文件写入失败")
            return None, "[ERROR] 音频文件写入失败", "\n".join(status_log)
        
        progress(1.0, desc="✅ 完成！")
        
        return output_path, f"✅ 合成完成！(时长: {duration:.2f}秒, 采样率: {sr} Hz)", "\n".join(status_log)
    except Exception as e:
        print(f"[ERROR] 合成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        status_log.append(f"❌ 错误: {str(e)}")
        return None, f"❌ 发生错误: {str(e)}", "\n".join(status_log)

# ═══════════════════════════════════════════════════
# 7. 构建 Gradio 界面 - 商用级设计 + 双语切换
# ═══════════════════════════════════════════════════

# 默认样式（保持 Gradio 原生风格）
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
}
"""

def build_voice_management_ui(lang="zh"):
    """构建音色管理界面"""
    t = TRANSLATIONS[lang]
    
    gr.Markdown(f"## {t['voice_mgmt_title']}")
    gr.Markdown(t['voice_mgmt_intro'])
    
    with gr.Tabs():
        # Tab 1: 音色列表
        with gr.Tab(t["tab_voice_list"]):
            gr.Markdown(f"### {t['voice_list_title']}")
            voices_table = gr.Markdown(format_voices_table())
            refresh_btn = gr.Button(f"🔄 {t['refresh_status']}", variant="secondary")
        
        # Tab 2: 创建音色
        with gr.Tab(t["tab_create_voice"]):
            gr.Markdown(f"### {t['create_voice_title']}")
            gr.Markdown(t['create_voice_steps'])
            
            with gr.Row():
                with gr.Column():
                    create_voice_name = gr.Textbox(
                        label=t["voice_name"],
                        placeholder=t["voice_name_example"],
                        info=t["voice_name_info"]
                    )
                    create_voice_audio = gr.Audio(
                        label=t["ref_audio"],
                        type="filepath"
                    )
                    create_voice_text = gr.Textbox(
                        label=t["ref_text"],
                        placeholder=t["ref_text_placeholder"],
                        lines=3,
                        info=t["ref_text_info"]
                    )
                    create_voice_btn = gr.Button(f"➕ {t['create_btn']}", variant="primary")
                
                with gr.Column():
                    create_status = gr.Textbox(
                        label=t["create_status"],
                        interactive=False,
                        lines=5
                    )
        
        # Tab 3: 编辑音色
        with gr.Tab(t["tab_edit_voice"]):
            gr.Markdown(f"### {t['edit_voice_title']}")
            gr.Markdown(t['edit_voice_intro'])
            
            with gr.Row():
                with gr.Column(scale=1):
                    edit_voice_selector = gr.Dropdown(
                        choices=_list_voices(),
                        label=t["select_voice"],
                        interactive=True
                    )
                    
                    edit_voice_info = gr.Markdown(t["please_select_voice"])
                    
                    edit_audio_player = gr.Audio(
                        label=t["current_ref_audio"],
                        type="filepath",
                        interactive=False
                    )
                    
                    with gr.Accordion(f"🔄 {t['update_audio_file']}", open=False):
                        new_audio_file = gr.Audio(
                            label=t["new_ref_audio"],
                            type="filepath"
                        )
                        update_audio_btn = gr.Button(f"🔄 {t['update_audio_btn']}", variant="secondary")
                
                with gr.Column(scale=1):
                    edit_ref_text = gr.Textbox(
                        label=t["ref_text_label"],
                        placeholder=t["ref_text_placeholder"],
                        lines=6
                    )
                    
                    gr.Markdown(t['edit_tips'])
                    
                    with gr.Row():
                        save_text_btn = gr.Button(f"💾 {t['save_ref_text']}", variant="primary")
                        clear_text_btn = gr.Button(f"🗑️ {t['clear_btn']}", variant="secondary")
                    
                    edit_status = gr.Textbox(
                        label=t["operation_status"],
                        interactive=False,
                        lines=3
                    )
        
        # Tab 4: 查看详情
        with gr.Tab(t["tab_view_details"]):
            gr.Markdown(f"### {t['view_details_title']}")
            
            with gr.Row():
                with gr.Column(scale=1):
                    detail_voice_selector = gr.Dropdown(
                        choices=_list_voices(),
                        label=t["select_voice"],
                        interactive=True
                    )
                    
                    detail_audio_player = gr.Audio(
                        label=t["ref_audio"],
                        type="filepath",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    detail_info = gr.Markdown(t["please_select_voice"])
                    
                    detail_ref_text = gr.Textbox(
                        label=t["ref_text_label"],
                        interactive=False,
                        lines=4
                    )
        
        # Tab 5: 删除音色
        with gr.Tab(t["tab_delete_voice"]):
            gr.Markdown(f"### {t['delete_voice_title']}")
            gr.Markdown(t['delete_warning'])
            
            with gr.Row():
                with gr.Column():
                    delete_voice_selector = gr.Dropdown(
                        choices=_list_voices(),
                        label=t["select_voice_to_delete"],
                        interactive=True
                    )
                    
                    delete_confirm = gr.Checkbox(
                        label=t["delete_confirm"],
                        value=False
                    )
                    
                    delete_voice_btn = gr.Button(f"🗑️ {t['delete_btn']}", variant="stop")
                
                with gr.Column():
                    delete_status = gr.Textbox(
                        label=t["delete_status"],
                        interactive=False,
                        lines=5
                    )
    
    # ========== 事件绑定 ==========
    
    # 刷新状态
    refresh_btn.click(
        fn=lambda: format_voices_table(lang),
        outputs=[voices_table]
    )
    
    # 创建音色
    create_voice_btn.click(
        fn=lambda *args: create_new_voice(*args, lang=lang),
        inputs=[create_voice_name, create_voice_audio, create_voice_text],
        outputs=[create_status, voices_table, edit_voice_selector]
    ).then(
        fn=lambda: gr.update(choices=_list_voices()),
        outputs=[delete_voice_selector, detail_voice_selector]
    )
    
    # 编辑音色 - 选择音色
    edit_voice_selector.change(
        fn=load_voice_for_edit,
        inputs=[edit_voice_selector],
        outputs=[edit_audio_player, edit_ref_text, edit_voice_info, edit_voice_selector]
    )
    
    # 编辑音色 - 保存参考文本
    save_text_btn.click(
        fn=lambda *args: save_voice_ref_text(*args, lang=lang),
        inputs=[edit_voice_selector, edit_ref_text],
        outputs=[edit_status, voices_table]
    )
    
    # 编辑音色 - 清空文本
    clear_text_btn.click(
        fn=lambda: ("", ""),
        outputs=[edit_ref_text, edit_status]
    )
    
    # 编辑音色 - 更新音频
    update_audio_btn.click(
        fn=lambda *args: update_voice_audio(*args, lang=lang),
        inputs=[edit_voice_selector, new_audio_file],
        outputs=[edit_status, voices_table]
    )
    
    # 查看详情
    detail_voice_selector.change(
        fn=get_voice_details,
        inputs=[detail_voice_selector],
        outputs=[detail_info, detail_ref_text, detail_audio_player, detail_voice_selector]
    )
    
    # 删除音色
    def safe_delete_voice(voice_name, confirmed):
        if not confirmed:
            return t["error_confirm_delete"], format_voices_table(lang), gr.update()
        return delete_voice_from_management(voice_name, lang=lang)
    
    delete_voice_btn.click(
        fn=safe_delete_voice,
        inputs=[delete_voice_selector, delete_confirm],
        outputs=[delete_status, voices_table, delete_voice_selector]
    ).then(
        fn=lambda: gr.update(choices=_list_voices()),
        outputs=[edit_voice_selector, detail_voice_selector]
    ).then(
        fn=lambda: False,
        outputs=[delete_confirm]
    )

def build_ui_tab(lang="zh"):
    """构建指定语言的界面标签页"""
    t = TRANSLATIONS[lang]
    
    with gr.Tabs():
        # 语音合成Tab
        with gr.Tab(t["tab_synthesis"]):
            with gr.Row():
                # 左栏：输入与设置
                with gr.Column(scale=3):
                    input_text = gr.Textbox(
                        label=t["input_label"],
                        placeholder=t["input_placeholder"],
                        lines=5,
                        value=t["input_default"]
                    )

                    voice_dropdown = gr.Dropdown(
                        choices=_list_voices(),
                        value="kazakh_man_1",
                        label=t["voice_preset"],
                        interactive=True
                    )

                    with gr.Accordion(t["create_voice"], open=False):
                        gr.Markdown(t["create_voice_warning"])
                        new_voice_name = gr.Textbox(
                            label=t["voice_name"], 
                            placeholder=t["voice_name_placeholder"]
                        )
                        new_voice_audio = gr.Audio(
                            label=t["ref_audio"], 
                            type="filepath"
                        )
                        new_voice_text = gr.Textbox(
                            label=t["ref_text"], 
                            placeholder=t["ref_text_placeholder"], 
                            lines=2
                        )
                        with gr.Row():
                            create_btn = gr.Button(t["save_voice"], variant="primary")
                            delete_btn = gr.Button(t["delete_voice"], variant="stop")
                        voice_status = gr.Textbox(label=t["operation_status"], interactive=False)

                    use_lora = gr.Checkbox(
                        label=t["enable_lora"], 
                        value=True
                    )
                    
                    with gr.Accordion(t["advanced_params"], open=False):
                        if lang == "zh":
                            gr.Markdown("""
**提示**: 
- 推理步数建议 5-15，过高会很慢
- CFG强度建议 1.5-3.0
- 英文文本建议使用较低参数以减少卡顿
                            """)
                        else:  # kk
                            gr.Markdown("""
**Кеңес**: 
- Қадам саны 5-15 ұсынылады, жоғары болса баяу
- CFG күші 1.5-3.0 ұсынылады
- Ағылшын мәтіні үшін төмен параметрлер ұсынылады
                            """)
                        
                        if lang == "zh":
                            timesteps_info = "推荐: 快速5-8, 标准10, 高质量15-20"
                            cfg_info = "推荐: 快速1.5-1.8, 标准2.0, 高质量2.5-3.0"
                        else:  # kk
                            timesteps_info = "Ұсыныс: жылдам 5-8, стандарт 10, жоғары сапа 15-20"
                            cfg_info = "Ұсыныс: жылдам 1.5-1.8, стандарт 2.0, жоғары сапа 2.5-3.0"
                        
                        timesteps = gr.Slider(
                            minimum=5, maximum=100, value=10, step=1, 
                            label=t["inference_steps"],
                            info=timesteps_info
                        )
                        cfg = gr.Slider(
                            minimum=1.0, maximum=5.0, value=2.0, step=0.1, 
                            label=t["cfg_strength"],
                            info=cfg_info
                        )
                        speed_rate = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0, step=0.1, 
                            label=t["speed"]
                        )
                        pitch_shift = gr.Slider(
                            minimum=-12, maximum=12, value=0, step=1, 
                            label=t["pitch"]
                        )
                        remove_silence_checkbox = gr.Checkbox(
                            label=t.get("remove_silence", "移除过长静音 (>0.8秒)" if lang == "zh" else "Ұзақ үнсіздікті жою (>0.8с)"),
                            value=True
                        )

                    with gr.Accordion(t["temp_voice_clone"], open=False):
                        gr.Markdown(t["temp_voice_warning"])
                        ref_audio = gr.Audio(
                            label=t["ref_audio"], 
                            type="filepath"
                        )
                        ref_text = gr.Textbox(
                            label=t["ref_text"], 
                            placeholder=t["ref_text_placeholder"],
                            lines=2
                        )

                    generate_btn = gr.Button(t["generate_btn"], variant="primary", size="lg")

                # 右栏：输出
                with gr.Column(scale=2):
                    output_audio = gr.Audio(label=t["audio_output"], type="filepath")
                    status_msg = gr.Textbox(label=t["status_info"], interactive=False, lines=2)
                    
                    progress_log = gr.Textbox(
                        label=t["inference_log"],
                        interactive=False,
                        lines=10,
                        placeholder=t["log_placeholder"]
                    )

                    gr.Markdown(t["usage_tips"])

            # ── 按钮绑定 ──
            create_btn.click(
                fn=lambda *args: create_voice(*args, lang=lang),
                inputs=[new_voice_name, new_voice_audio, new_voice_text],
                outputs=[voice_dropdown, voice_status]
            )

            delete_btn.click(
                fn=lambda *args: delete_voice(*args, lang=lang),
                inputs=[voice_dropdown],
                outputs=[voice_dropdown, voice_status]
            )

            generate_btn.click(
                fn=lambda *args: tts_generate(*args, lang=lang),
                inputs=[input_text, voice_dropdown, ref_audio, ref_text,
                        use_lora, timesteps, cfg, speed_rate, pitch_shift, remove_silence_checkbox],
                outputs=[output_audio, status_msg, progress_log]
            )
        
        # 音色管理Tab
        with gr.Tab(t["tab_voice_management"]):
            build_voice_management_ui(lang)


with gr.Blocks(title="多语言 TTS 引擎 / Көптілді TTS жүйесі", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 多语言 TTS 引擎 / Көптілді TTS жүйесі")
    gr.Markdown("支持哈萨克语、中文、英文及多语言混合 · 零样本音色克隆 · 实时语音合成")
    gr.Markdown("Қазақ, қытай, ағылшын тілдерін қолдайды · Үлгісіз дауыс клондау · Нақты уақыт режимінде сөйлеу синтезі")
    
    with gr.Tabs():
        with gr.Tab("中文 Chinese"):
            build_ui_tab("zh")
        
        with gr.Tab("Қазақша Kazakh"):
            build_ui_tab("kk")
    
    gr.Markdown("---")
    gr.Markdown("""
<div style="text-align: center; color: #666;">
    <p>多语言 TTS 引擎 - 哈萨克语强化版 / Көптілді TTS жүйесі - Қазақ тілі нұсқасы</p>
</div>
    """)

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
