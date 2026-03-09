import base64
from openai import OpenAI

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

def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# 确保视频路径正确
video_path = "./test_real_videos.mp4"
base64_video = encode_video(video_path)

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

print(response.choices[0].message.content)
with open("real_videos_result.txt", "w", encoding="utf-8") as f:
    f.write(response.choices[0].message.content)