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
你是一名专业的数字视频取证专家，擅长识别人脸深度伪造痕迹。现在你将接收一段明确被篡改的视频片段。请仔细观察画面，并基于画面中可直接观察到的视觉异常，给出具由说服力的伪造证据说明。
请注意：1.仅描述可以从画面中直接观察到的异常现象。2.不得假设音频或跨模态信息。3.不要输出不明显或不确定的异常。4.只选择最显著、最具判别力的1到3条证据。
可选观察维度（从中选择最相关的进行分析，不必全部涉及，也可选择其他你认为存在伪造的维度）：
眼部细节异常：瞳孔形状是否异常，眨眼是否不自然。
边缘融合异常：面部轮廓、下颌线、发际线或脸与背景交界处是否存在模糊、重影、锯齿或拼接痕迹。
表情与肌肉运动不自然：面部运动是否僵硬、滞后或缺乏细微肌肉变化，表情过渡是否不连续。
几何结构异常：五官比例是否出现轻微错位、缩放不协调、局部变形或透视关系异常。
局部纹理不一致：面部皮肤纹理是否与颈部或背景分辨率不一致，是否出现异常平滑或局部噪声增强。
口腔与牙齿细节异常：牙齿纹理模糊、边界闪烁、口腔内部结构在运动时发生形变或不连续。
伪影异常：面部或局部区域是否出现块状失真、纹理断裂、闪烁噪点、局部模糊块。
光照与阴影异常：面部光照方向的阴影投射是否异常，是否存在无法解释的高光或阴影缺失。
输出要求：1.仅输出最具判别力的1到3条证据。2.不要重复类似证据。3.不要输出不明显或推测性内容，1条明显证据质量大于3条不明显证据。4.输出必须使用以下结构化格式(不要输出任何空格)：
结论：该视频是伪造的。证据1类别：（从上述维度中选择最贴切的类别名称）.异常描述：.证据2（如有）类别：.异常描述：.证据3（如有）类别：.异常描述：.
"""


INPUT_JSON = "./fv_ra.json"
TRAIN_JSON = "./fv_ra_meta.json"

TRAIN_DIR = "D:/FIQD/train/fake/V"
VAL_DIR = "D:/FIQD/val/fake/V"
TEST_DIR = "D:/FIQD/test/fake/V"

QUESTION = "这是一条已经确认为伪造的视频，请给出几条判定依据。"
LABEL = "fake_video_real_audio"

MAX_DATA = 455
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
fv_ra_meta_data = []
for idx in range(0, len(dataset), 8):
    if count > MAX_DATA:
        break

    item = dataset[idx]

    
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

    if count <= 350:

        save_dir = TRAIN_DIR

    elif count <= 385:

        save_dir = VAL_DIR

    else:

        save_dir = TEST_DIR

    save_path = f"{save_dir}/fv_ra_{count}.mp4"
    sample_id = f"fv_ra_{count}"

    # =========================
    # 保存视频
    # =========================
    try:

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(clip_len),
            "-i", video_path,
            "-an",                # 删除音频
            "-c:v", "libx264",    # 视频编码

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

    fv_ra_meta_data.append(entry)

    print(f"成功生成: {count}")


    # =========================
    # 定期保存
    # =========================

    if count % SAVE_INTERVAL == 0:

        with open(TRAIN_JSON, "w", encoding="utf-8") as f:
            json.dump(fv_ra_meta_data, f, ensure_ascii=False, indent=4)
    
        print("已自动保存 fv_ra_meta.json")

    count += 1

# =========================
# 任务完成
# =========================

print("=================================")
print(f"任务全部完成！")
print(f"fv_ra_meta.json文件中已经包含 {len(fv_ra_meta_data)} 条数据！")
print("=================================")