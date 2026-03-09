import base64
from openai import OpenAI


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
def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


base64_audio = encode_audio("./test_audio1.wav")

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
print(response.choices[0].message.content)
with open("audio_result.txt", "w", encoding="utf-8") as f:
    f.write(response.choices[0].message.content)
