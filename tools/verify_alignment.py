"""
验证参考文本和音频是否对齐
"""
import os
import json
import torchaudio

def verify_voice(voice_name):
    """验证音色的对齐情况"""
    wav_path = f"voices/{voice_name}/ref.wav"
    meta_path = f"voices/{voice_name}/meta.json"
    
    if not os.path.exists(wav_path):
        return None
    
    # 读取音频
    audio, sr = torchaudio.load(wav_path)
    duration = audio.shape[1] / sr
    
    # 读取元数据
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    ref_text = meta.get("ref_text", "")
    
    # 计算指标
    word_count = len(ref_text.split()) if ref_text else 0
    estimated_capacity = duration * 2.5  # 每秒约2.5词
    
    # 判断对齐情况
    if not ref_text:
        status = "❌ 空文本"
        score = 0
    elif word_count > estimated_capacity * 1.3:
        status = "❌ 文本过长"
        score = 30
    elif word_count < estimated_capacity * 0.4:
        status = "⚠️ 文本过短"
        score = 60
    elif word_count > estimated_capacity * 1.1:
        status = "⚠️ 文本略长"
        score = 80
    elif word_count < estimated_capacity * 0.6:
        status = "⚠️ 文本略短"
        score = 80
    else:
        status = "✅ 对齐良好"
        score = 100
    
    return {
        "voice": voice_name,
        "duration": duration,
        "word_count": word_count,
        "estimated_capacity": estimated_capacity,
        "ratio": word_count / estimated_capacity if estimated_capacity > 0 else 0,
        "status": status,
        "score": score,
        "ref_text": ref_text
    }

def main():
    print("=" * 80)
    print("参考文本对齐验证")
    print("=" * 80)
    
    # 检查所有音色
    voices = []
    if os.path.exists("voices"):
        voices = [d for d in os.listdir("voices") if os.path.isdir(f"voices/{d}")]
    
    if not voices:
        print("❌ 未找到音色文件")
        return
    
    results = []
    for voice in sorted(voices):
        result = verify_voice(voice)
        if result:
            results.append(result)
    
    # 显示结果
    print(f"\n{'音色':<20} {'时长':<8} {'词数':<6} {'容量':<6} {'比例':<8} {'状态':<15} {'评分'}")
    print("-" * 90)
    
    for r in results:
        ratio_str = f"{r['ratio']:.2f}"
        print(f"{r['voice']:<20} {r['duration']:<8.2f} {r['word_count']:<6} {r['estimated_capacity']:<6.1f} {ratio_str:<8} {r['status']:<15} {r['score']}")
    
    # 统计
    print("\n" + "=" * 80)
    print("统计")
    print("=" * 80)
    
    perfect = [r for r in results if r['score'] == 100]
    good = [r for r in results if 80 <= r['score'] < 100]
    warning = [r for r in results if 50 <= r['score'] < 80]
    bad = [r for r in results if r['score'] < 50]
    
    print(f"✅ 对齐良好: {len(perfect)} 个")
    print(f"⚠️ 需要注意: {len(good)} 个")
    print(f"⚠️ 可能有问题: {len(warning)} 个")
    print(f"❌ 严重问题: {len(bad)} 个")
    
    # 显示问题音色
    if bad or warning:
        print("\n" + "=" * 80)
        print("需要修复的音色")
        print("=" * 80)
        
        for r in bad + warning:
            print(f"\n{r['voice']} ({r['status']}):")
            print(f"  音频时长: {r['duration']:.2f}秒")
            print(f"  参考文本: {r['ref_text'][:60]}...")
            print(f"  词数: {r['word_count']} (建议: {r['estimated_capacity']:.0f})")
            
            if r['score'] == 0:
                print(f"  建议: 使用 manual_align_text.py 手动输入参考文本")
            elif r['word_count'] > r['estimated_capacity'] * 1.3:
                print(f"  建议: 参考文本过长，需要缩短或重新录制音频")
            elif r['word_count'] < r['estimated_capacity'] * 0.4:
                print(f"  建议: 参考文本过短，可能遗漏了部分内容")
    
    print("\n" + "=" * 80)
    print("对齐标准")
    print("=" * 80)
    print("""
理想比例: 0.8 - 1.1 (参考文本词数 / 估计容量)

评分标准:
  100分: 比例在 0.6-1.1 之间，对齐良好
   80分: 比例在 0.4-0.6 或 1.1-1.3 之间，略有偏差
   60分: 比例在 0.4 以下，文本过短
   30分: 比例在 1.3 以上，文本过长
    0分: 参考文本为空

修复方法:
  1. 自动识别: python transcribe_audio.py (需要安装 Whisper)
  2. 手动对齐: python manual_align_text.py
  3. 重新录制: 录制新的参考音频（3-5秒，清晰）
""")

if __name__ == "__main__":
    main()
