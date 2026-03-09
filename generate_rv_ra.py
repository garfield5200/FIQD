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
你是一名专业的多模态数字取证专家，负责对视频内容进行真实性核查。现在你将接收到一段明确为真实的视频。你的任务是基于可直接观察和听到的自然特征，列举那些符合真实规律和特征的细节。
请注意：1.仅描述可以从视频中直接看到或听到的自然一致性特征。2.不进行夸张或文学化表达。3.只选择最显著、最符合真实规律的1到3条细节。
可选观察维度（从中选择最真实的维度进行分析，不必全部涉及，也可选择其他你认为真实的维度）：
自然的生理特征：人物说话语气和动作、眨眼动作、表情变化等生理细节自然流畅。
面部与边缘自然连续性：面部轮廓与背景交界自然，无异常模糊、拼接或局部失真。
语调与韵律自然：说话人的语调自然真实，符合人类自然的情感起伏；韵律节奏符合正常说话，停顿位置符合语言习惯。
呼吸细节正常：存在自然呼吸声、吞咽声或微小口腔噪音。
光照与阴影物理一致性：光源方向稳定，阴影变化与头部或物体运动逻辑保持一致。
细节纹理一致性：皮肤纹理、发丝、衣物褶皱等高频细节清晰且连续，无异常平滑或块状失真。
时空连贯性：面部或物体在运动过程中保持帧间连续。
唇音同步性：说话人的嘴型与发音内容相一致。
输出要求：1.仅输出最显著、最符合真实规律的1到3条细节。2.不要重复输出类似表述。3.不要输出不明显或推测性内容，1条明显细节质量大于3条不明显细节。4.输出必须使用以下结构化格式(不要输出任何空格)：
结论：该视频是真实的。细节1类别：（从上述维度中选择最贴切的类别名称）。细节描述：。细节2（如有）类别：。细节描述：。细节3（如有）类别：。细节描述：。
"""

INPUT_JSON = "./rv_ra.json"
TRAIN_JSON = "./rv_ra_meta.json"

TRAIN_DIR = "D:/FIQD/train/real"
VAL_DIR = "D:/FIQD/val/real"
TEST_DIR = "D:/FIQD/test/real"

QUESTION = "这是一条已经确认为真实的视频，请给出几条判定依据。"
LABEL = "real_video_real_audio"

MAX_DATA = 325
SAVE_INTERVAL = 5

#函数
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def get_video_duration(video_path):

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
rv_ra_meta_data = []
for idx, item in enumerate(dataset):

    if count > MAX_DATA:
        break

   
    video_path = "D:/" + item["new_path"]

    if not os.path.exists(video_path):
        print("视频不存在:", video_path)
        continue

    # =========================
    # 读取视频
    # =========================

    try:

        duration = get_video_duration(video_path)

        if duration > 4:

            clip_len = random.choice([2, 3])
            start = 0

        else:

            clip_len = duration
            start = 0

    except Exception as e:

        print("读取视频长度失败:", e)
        continue
     # =========================
    # 确定保存路径
    # =========================

    if count <= 250:

        save_dir = TRAIN_DIR

    elif count <= 275:

        save_dir = VAL_DIR

    else:

        save_dir = TEST_DIR

    save_path = f"{save_dir}/rv_ra_{count}.mp4"
    sample_id = f"rv_ra_{count}"

    # =========================
    # 保存视频
    # =========================
    try:

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(clip_len),
            "-c:v", "libx264",
            "-c:a", "aac",
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
        base64_video = encode_video(save_path)
        response = client.chat.completions.create(
            model="gemini-3-pro-preview", 
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:video/mp4;base64,{base64_video}" 
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

    rv_ra_meta_data.append(entry)

    print(f"成功生成: {count}")


    # =========================
    # 定期保存
    # =========================

    if count % SAVE_INTERVAL == 0:

        with open(TRAIN_JSON, "w", encoding="utf-8") as f:
            json.dump(rv_ra_meta_data, f, ensure_ascii=False, indent=4)
    
        print("已自动保存 rv_ra_meta.json")

    count += 1

# =========================
# 任务完成
# =========================

print("=================================")
print(f"任务全部完成！")
print(f"rv_ra_meta.json文件中已经包含 {len(rv_ra_meta_data)} 条数据！")
print("=================================")