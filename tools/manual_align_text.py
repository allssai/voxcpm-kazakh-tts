"""
手动对齐音频和参考文本
"""
import os
import json
import torchaudio

def show_voice_info(voice_name):
    """显示音色信息"""
    wav_path = f"voices/{voice_name}/ref.wav"
    meta_path = f"voices/{voice_name}/meta.json"
    
    if not os.path.exists(wav_path):
        print(f"❌ 音频文件不存在: {wav_path}")
        return None
    
    # 读取音频信息
    audio, sr = torchaudio.load(wav_path)
    duration = audio.shape[1] / sr
    
    # 读取元数据
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    ref_text = meta.get("ref_text", "")
    original_ref_text = meta.get("_original_ref_text", "")
    
    print(f"\n音色: {voice_name}")
    print(f"音频: {duration:.2f}秒, {sr}Hz, {audio.shape[0]}声道")
    print(f"当前参考文本: {ref_text if ref_text else '(空)'}")
    if original_ref_text:
        print(f"原始参考文本: {original_ref_text}")
    
    # 估计能容纳多少词
    estimated_words = duration * 2.5
    actual_words = len(ref_text.split()) if ref_text else 0
    
    print(f"\n估计容量: {estimated_words:.1f} 词")
    print(f"当前文本: {actual_words} 词")
    
    if ref_text and actual_words > estimated_words * 1.2:
        print("⚠️ 警告: 参考文本可能过长")
    elif not ref_text:
        print("⚠️ 警告: 参考文本为空")
    else:
        print("✓ 参考文本长度合理")
    
    return meta

def update_ref_text(voice_name, new_text):
    """更新参考文本"""
    meta_path = f"voices/{voice_name}/meta.json"
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    old_text = meta.get("ref_text", "")
    
    # 备份旧文本
    if old_text:
        meta["_old_ref_text"] = old_text
    
    # 更新
    meta["ref_text"] = new_text
    meta["_manually_aligned"] = True
    meta["_note"] = "参考文本已手动对齐"
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已更新参考文本")

def main():
    print("=" * 80)
    print("手动对齐音频和参考文本")
    print("=" * 80)
    
    voices = ["english_man_1", "myself", "trump"]
    
    print("\n有问题的音色:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")
    
    print("\n步骤:")
    print("1. 选择一个音色")
    print("2. 播放音频，听清楚说的内容")
    print("3. 输入音频中实际说的内容（必须100%准确）")
    print("4. 保存并测试")
    
    while True:
        print("\n" + "=" * 80)
        choice = input("\n选择音色 (1-3) 或输入 q 退出: ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(voices):
                print("❌ 无效选择")
                continue
        except ValueError:
            print("❌ 无效输入")
            continue
        
        voice = voices[idx]
        meta = show_voice_info(voice)
        
        if not meta:
            continue
        
        print("\n" + "-" * 80)
        print("请播放音频文件，听清楚内容:")
        print(f"  voices/{voice}/ref.wav")
        print("\n提示:")
        print("  - 可以使用 Windows Media Player 或其他播放器")
        print("  - 多听几遍，确保听清楚每个词")
        print("  - 参考文本必须与音频内容100%一致")
        print("-" * 80)
        
        input("\n听完后按 Enter 继续...")
        
        print("\n请输入音频中实际说的内容:")
        print("(必须100%准确，包括标点符号)")
        new_text = input("> ").strip()
        
        if not new_text:
            print("❌ 参考文本不能为空")
            continue
        
        # 显示确认
        print(f"\n新的参考文本: {new_text}")
        print(f"词数: {len(new_text.split())}")
        
        confirm = input("\n确认更新？(y/n): ").strip().lower()
        
        if confirm == 'y':
            update_ref_text(voice, new_text)
            print("\n✅ 更新成功！")
            print("\n建议:")
            print("1. 重启应用: python web_app.py")
            print("2. 测试这个音色")
            print("3. 如果还有问题，重新对齐")
        else:
            print("已取消")
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
