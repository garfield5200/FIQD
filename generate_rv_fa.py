import json
import os
import random
import base64
from openai import OpenAI
import subprocess

# =========================
# 配置
# =========================
client = OpenAI(api_key="", base_url="https://hiapi.online/v1")

prompt ="""
你是一名专业的数字音频取证专家，专注于识别音频篡改痕迹。现在你将接收一段明确被篡改的音频片段。请仔细分析该音频，并基于音频中可直接听到或可分析到的异常现象，给出具有说服力的伪造证据说明。
请注意：1.仅描述可以从音频中的异常现象。2.不得假设视频或跨模态信息。3.只选择最显著、最具判别力的1到3条证据。
可选观察维度（从中选择最相关的进行分析，不必全部涉及，也可选择其他你认为伪造的维度）：
背景噪声一致性异常：背景中是否存在非自然的电子底噪、机械电流声，或者背景噪声在语音起止处是否发生突变。
语调与韵律异常：评估说话人的语调是否生硬、平淡，缺乏人类自然的情感起伏；韵律节奏是否机械化，停顿位置是否违反语言习惯。
发音瞬态失真：辅音（如爆破音、擦音）是否模糊、削弱或失真，瞬态细节是否缺失。
音色稳定性异常：说话人音色是否在连续语音中发生不合理变化。
音量异常：语音的音量包络是否平滑，是否存在异常的突增或突降。
呼吸与生理细节缺失：是否完全缺失自然呼吸声、吞咽声或微小口腔噪音。
输出要求：1.仅输出最具判别力的1到3条证据。2.不要重复类似证据。3.不要输出不明显或推测性内容，1条明显证据质量大于3条不明显证据。4.输出必须使用以下结构化格式(不要输出任何空格)：
结论：该音频是伪造的。证据1类别：（从上述维度中选择最贴切的类别名称）.异常描述：.证据2（如有）类别：.异常描述：.证据3（如有）类别：.异常描述：.
"""

INPUT_JSON = "./rv_fa.json"
TRAIN_JSON = "./rv_fa_meta.json"

TRAIN_DIR = "D:/FIQD/train/fake/A"
VAL_DIR = "D:/FIQD/val/fake/A"
TEST_DIR = "D:/FIQD/test/fake/A"

QUESTION = "这是一条已经确认为伪造的音频，请给出几条判定依据。"
LABEL = "real_video_fake_audio"

MAX_DATA = 260
SAVE_INTERVAL = 5

#函数
def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def get_audio_duration(video_path):
    """获取视频中的音频长度"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())
  
# =========================
# 创建目录
# =========================

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# =========================
# 读取train.json (断点续跑)
# =========================



count = 1

print(f"当前已有数据: {count-1}")

# =========================
# 读取dataset
# =========================

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"待处理数据: {len(dataset)}")

# =========================
# 主循环
# =========================
rv_fa_meta_data = []
for idx, item in enumerate(dataset):

    if count > MAX_DATA:
        break

    
    video_path = "D:/" + item["new_path"]

    if not os.path.exists(video_path):
        print("视频不存在:", video_path)
        continue

    # =========================
    # 读取音频
    # =========================

    try:
        duration = get_audio_duration(video_path)

        if duration > 4:
            clip_len = random.choice([2, 3])
            start = 0
        else:
            clip_len = duration
            start = 0

    except Exception as e:
        print("读取音频长度失败:", e)
        continue

     # =========================
    # 确定保存路径
    # =========================

    if count <= 200:

        save_dir = TRAIN_DIR

    elif count <= 220:

        save_dir = VAL_DIR

    else:

        save_dir = TEST_DIR

    save_path = f"{save_dir}/rv_fa_{count}.wav"
    sample_id = f"rv_fa_{count}"

    # =========================
    # 保存音频
    # =========================
    try:

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(clip_len),

            "-vn",                 # 不要视频
            "-acodec", "pcm_s16le",# wav编码
            "-ar", "16000",        # 16k采样率
            "-ac", "1",            # 单声道

            "-loglevel", "quiet",
            save_path
        ]

        subprocess.run(cmd, check=True)

    except Exception as e:

        print("ffmpeg裁剪失败:", e)
        continue

   
    # =========================
    # 调用大模型
    # =========================

    try:
        base64_audio = encode_audio(save_path)
        response = client.chat.completions.create(
            model="gemini-3-pro-preview", 
            messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": f"data:;base64,{base64_audio}",
                                    "format": "wav",
                                },
                            },
                        ],
                    },
                ],
            stream=False,
        )
        output = response.choices[0].message.content

    except Exception as e:

        print("大模型调用失败:", e)
        continue

 
    # =========================
    # 构建数据条目
    # =========================

    entry = {
        "path": save_path,
        "id": sample_id,
        "label": LABEL,
        "question": QUESTION,
        "answer": output
    }

    rv_fa_meta_data.append(entry)

    print(f"成功生成: {count}")


    # =========================
    # 定期保存
    # =========================

    if count % SAVE_INTERVAL == 0:

        with open(TRAIN_JSON, "w", encoding="utf-8") as f:
            json.dump(rv_fa_meta_data, f, ensure_ascii=False, indent=4)
    
        print("已自动保存 fv_fa_meta.json")

    count += 1

# =========================
# 任务完成
# =========================

print("=================================")
print(f"任务全部完成！")
print(f"rv_fa_meta.json文件中已经包含 {len(rv_fa_meta_data)} 条数据！")
print("=================================")