"""
最终修复方案：
1. 修复音频格式
2. 手动设置正确的参考文本（基于音频实际内容）
"""
import torchaudio
import torch
import json
import os

def fix_and_align_voice(voice_name, correct_ref_text, target_duration=5.0):
    """
    修复音频并设置正确的参考文本
    
    Args:
        voice_name: 音色名称
        correct_ref_text: 正确的参考文本（与音频内容100%匹配）
        target_duration: 目标时长（秒）
    """
    voice_dir = f"voices/{voice_name}"
    wav_path = os.path.join(voice_dir, "ref.wav")
    meta_path = os.path.join(voice_dir, "meta.json")
    
    print(f"\n{'='*60}")
    print(f"修复音色: {voice_name}")
    print(f"{'='*60}")
    
    # 1. 修复音频
    audio, sr = torchaudio.load(wav_path)
    print(f"原始音频: {audio.shape[0]}声道, {sr}Hz, {audio.shape[1]/sr:.2f}秒")
    
    modified = False
    
    # 转换为单声道
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        print(f"✓ 转换为单声道")
        modified = True
    
    # 重采样到 44100
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)
        sr = 44100
        print(f"✓ 重采样到 44100Hz")
        modified = True
    
    # 截取到目标时长
    target_samples = int(target_duration * sr)
    if audio.shape[1] > target_samples:
        start_idx = (audio.shape[1] - target_samples) // 2
        audio = audio[:, start_idx:start_idx + target_samples]
        print(f"✓ 截取到 {target_duration}秒")
        modified = True
    
    if modified:
        torchaudio.save(wav_path, audio, sr)
        print(f"✓ 音频已保存")
    
    # 2. 更新参考文本
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    old_ref_text = meta.get("ref_text", "")
    
    meta["ref_text"] = correct_ref_text
    meta["_old_ref_text"] = old_ref_text
    meta["_manually_fixed"] = True
    meta["_note"] = "音频和参考文本已手动对齐"
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 参考文本已更新")
    print(f"  旧: {old_ref_text[:40]}...")
    print(f"  新: {correct_ref_text}")
    
    # 3. 验证
    final_audio, final_sr = torchaudio.load(wav_path)
    final_duration = final_audio.shape[1] / final_sr
    word_count = len(correct_ref_text.split())
    estimated_capacity = final_duration * 2.5
    ratio = word_count / estimated_capacity
    
    print(f"\n验证:")
    print(f"  音频: {final_duration:.2f}秒, {final_audio.shape[0]}声道, {final_sr}Hz")
    print(f"  文本: {word_count}词")
    print(f"  容量: {estimated_capacity:.1f}词")
    print(f"  比例: {ratio:.2f}")
    
    if 0.6 <= ratio <= 1.1:
        print(f"  ✅ 对齐良好")
        return True
    elif 0.4 <= ratio < 0.6 or 1.1 < ratio <= 1.3:
        print(f"  ⚠️ 略有偏差，但可接受")
        return True
    else:
        print(f"  ❌ 对齐不佳，建议重新调整")
        return False

def main():
    print("=" * 80)
    print("最终修复方案")
    print("=" * 80)
    print("""
这个脚本将：
1. 修复音频格式（单声道, 44100Hz, 5秒）
2. 设置正确的参考文本（基于音频实际内容）

重要提示：
- 参考文本必须与音频内容100%匹配
- 如果不确定音频内容，请先播放音频听清楚
- 参考文本应该是音频中实际说的内容，不能多也不能少
""")
    
    # 定义修复方案
    fixes = [
        {
            "voice": "english_man_1",
            "ref_text": "Welcome to the multilingual text to speech system.",
            "note": "假设音频说的是这个（需要你确认）"
        },
        {
            "voice": "myself",
            "ref_text": "дауысыңды да ұрлай аламын. Мұны қасыңдағы адамдарға, әсіресе қарттарға.",
            "note": "假设音频说的是中间部分（需要你确认）"
        },
        {
            "voice": "trump",
            "ref_text": "I like him a lot. I have a great relationship with him.",
            "note": "假设音频说的是后半部分（需要你确认）"
        }
    ]
    
    print("\n将修复以下音色:")
    for fix in fixes:
        print(f"\n{fix['voice']}:")
        print(f"  参考文本: {fix['ref_text']}")
        print(f"  说明: {fix['note']}")
    
    print("\n" + "=" * 80)
    print("⚠️ 重要：请先播放音频，确认参考文本是否正确！")
    print("=" * 80)
    print("\n音频文件位置:")
    for fix in fixes:
        print(f"  voices/{fix['voice']}/ref.wav")
    
    choice = input("\n确认参考文本正确后，输入 y 继续，其他键退出: ").strip().lower()
    
    if choice != 'y':
        print("\n已取消。")
        print("\n如果参考文本不正确，请:")
        print("1. 播放音频，听清楚内容")
        print("2. 运行: python manual_align_text.py")
        print("3. 手动输入正确的参考文本")
        return
    
    # 执行修复
    results = []
    for fix in fixes:
        success = fix_and_align_voice(fix['voice'], fix['ref_text'])
        results.append((fix['voice'], success))
    
    # 总结
    print("\n" + "=" * 80)
    print("修复完成")
    print("=" * 80)
    
    for voice, success in results:
        status = "✅ 成功" if success else "⚠️ 需要调整"
        print(f"{voice}: {status}")
    
    print("\n下一步:")
    print("1. 重启应用: python web_app.py")
    print("2. 测试这些音色")
    print("3. 如果还有问题，运行: python verify_alignment.py")

if __name__ == "__main__":
    main()
