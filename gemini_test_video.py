import base64
from openai import OpenAI

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

def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# 确保视频路径正确
video_path = "./test_video1.mp4"
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
with open("video_result.txt", "w", encoding="utf-8") as f:
    f.write(response.choices[0].message.content)