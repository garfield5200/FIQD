import base64
from openai import OpenAI

client = OpenAI(api_key="", base_url="https://hiapi.online/v1")


prompt ="""
你是一名专业的多模态数字取证专家，擅长识别视觉与听觉模态中的伪造痕迹及其跨模态不一致现象。现在你将接收一段明确被篡改的视频。你的任务是基于可直接观察和听到的异常现象，给出具有说服力的伪造证据说明。
请注意：1.仅描述可以从视频中直接看到或到到的的异常现象。2.可跨模态选择最强证据，不要求每个模态都必须出现。3.只选择最显著、最具判别力的1到3条证据。
可选观察维度（从中选择最相关的进行分析，不必全部涉及，也可选择其他你认为伪造的维度）：
视觉模态：
眼部细节异常：瞳孔形状是否异常，眨眼是否不自然。
边缘融合异常：面部轮廓、下颌线、发际线或脸与背景交界处是否存在模糊、重影、锯齿或拼接痕迹。
表情与肌肉运动不自然：面部运动是否僵硬、滞后或缺乏细微肌肉变化，表情过渡是否不连续。
口腔与牙齿细节异常：牙齿纹理模糊、边界闪烁、口腔内部结构在运动时发生形变或不连续。
伪影异常：面部或局部区域是否出现块状失真、纹理断裂、闪烁噪点、局部模糊块。
听觉模态：
背景噪声一致性异常：背景中是否存在非自然的电子底噪、机械电流声，或者背景噪声在语音起止处是否发生突变。
语调与韵律异常：评估说话人的语调是否生硬、平淡，缺乏人类自然的情感起伏；韵律节奏是否机械化，停顿位置是否违反语言习惯。
发音瞬态失真：辅音（如爆破音、擦音）是否模糊、削弱或失真，瞬态细节是否缺失。
跨膜态：
唇音同步性：说话人的嘴型是否与发音内容相一致。
情感匹配性：面部表情传达的情感与语音语调传达的情感是否一致。
身份一致性：声音音色、年龄感、性别特征是否与画面中人物的外貌特征相匹配。
环境一致性：视觉背景与听觉背景是否矛盾。
输出要求：1.仅输出最具判别力的1到3条证据。2.不要重复类似证据。3.不要输出不明显或推测性内容，1条明显证据质量大于3条不明显证据。4.输出必须使用以下结构化格式(不要输出任何空格)：
结论：该视频是伪造的。证据1类别：（从上述维度中选择最贴切的类别名称）.异常描述：.证据2（如有）类别：.异常描述：.证据3（如有）类别：.异常描述：.
"""

def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# 确保视频路径正确
video_path = "./test_videos1.mp4"
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
with open("videos_result.txt", "w", encoding="utf-8") as f:
    f.write(response.choices[0].message.content)