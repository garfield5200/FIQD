import json
import os
import subprocess
import base64
from openai import OpenAI

# =========================
# 配置
# =========================

INPUT_JSON = "./metadata.min.json"

DATA_ROOT = "D:/LAV-DF/LAV-DF"
SAVE_ROOT = "D:/FIQD_LAV-DF"

os.makedirs(SAVE_ROOT, exist_ok=True)

OUTPUT_JSON = "./fiqd_lavdf_meta.json"

client = OpenAI(api_key="", base_url="https://hiapi.online/v1")

# =========================
# prompt
# =========================

PROMPT_RV_FA ="""
你是一名专业的数字音频取证专家，专注于识别音频篡改痕迹。现在你将接收一段明确被篡改的音频片段。请仔细分析该音频，并基于音频中可直接听到或可分析到的异常现象，给出具有说服力的伪造证据说明。
请注意：1.仅描述可以从音频中的异常现象。2.不得假设视频或跨模态信息。3.只选择最显著、最具判别力的1到3条证据。
可选观察维度（从中选择最相关的进行分析，不必全部涉及，也可选择其他你认为伪造的维度）：
背景噪声一致性异常：背景中是否存在非自然的电子底噪、机械电流声，或者背景噪声在语音起止处是否发生突变。
语调与韵律：评估说话人的语调是否生硬、平淡，缺乏人类自然的情感起伏；韵律节奏是否机械化，停顿位置是否违反语言习惯。
发音瞬态失真：辅音（如爆破音、擦音）是否模糊、削弱或失真，瞬态细节是否缺失。
音色稳定性异常：说话人音色是否在连续语音中发生不合理变化。
音量异常：语音的音量包络是否平滑，是否存在异常的突增或突降。
呼吸与生理细节缺失：是否完全缺失自然呼吸声、吞咽声或微小口腔噪音。
输出要求：1.仅输出最具判别力的1到3条证据。2.不要重复类似证据。3.不要输出不明显或推测性内容，1条明显证据质量大于3条不明显证据。4.输出必须使用以下结构化格式(不要输出任何空格)：
结论：该音频是伪造的。证据1类别：（从上述维度中选择最贴切的类别名称）.异常描述：.证据2（如有）类别：.异常描述：.证据3（如有）类别：.异常描述：.
"""
PROMPT_RV_RA ="""
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
PROMPT_FV_FA ="""
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
PROMPT_FV_RA ="""
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
QUESTION_FV_FA = "这是一条已经确认为伪造的视频，请给出几条判定依据。"
QUESTION_FV_RA = QUESTION_FV_FA
QUESTION_RV_FA = "这是一条已经确认为伪造的音频，请给出几条判定依据。"
QUESTION_RV_RA = "这是一条已经确认为真实的视频，请给出几条判定依据。"

# =========================
# 计数器
# =========================

target = {
    "fv_fa":20,
    "fv_ra":40,
    "rv_fa":20,
    "rv_ra":20
}

count = {
    "fv_fa":0,
    "fv_ra":0,
    "rv_fa":0,
    "rv_ra":0
}

meta = []

# =========================
# 工具函数
# =========================

def encode_video(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

def encode_audio(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

def cut_video(src,start,dur,out):
    cmd=[
        "ffmpeg","-y",
        "-ss",str(start),
        "-i",src,
        "-t",str(dur),
        "-c:v","libx264",
        "-c:a","aac",
        "-loglevel","quiet",
        out
    ]
    subprocess.run(cmd)

def cut_video_visual_only(src,start,dur,out):

    cmd=[
        "ffmpeg",
        "-y",
        "-ss",str(start),
        "-i",src,
        "-t",str(dur),
        "-an",
        "-c:v","libx264",
        "-loglevel","quiet",
        out
    ]

    subprocess.run(cmd)

def cut_audio(src,start,dur,out):
    cmd=[
        "ffmpeg","-y",
        "-ss",str(start),
        "-i",src,
        "-t",str(dur),
        "-vn",
        "-acodec","pcm_s16le",
        "-ar","16000",
        "-ac","1",
        "-loglevel","quiet",
        out
    ]
    subprocess.run(cmd)

# =========================
# 读取数据
# =========================

with open(INPUT_JSON,"r",encoding="utf-8") as f:
    dataset=json.load(f)

# =========================
# 主循环
# =========================

for item in dataset:

    if all(count[k] >= target[k] for k in target):
        break

    path = os.path.join(DATA_ROOT,item["file"])

    if not os.path.exists(path):
        continue

    modify_video = item["modify_video"]
    modify_audio = item["modify_audio"]
    n_fakes = item["n_fakes"]
    fake_periods = item["fake_periods"]

    # ======================
    # fv_fa
    # ======================

    if modify_video and modify_audio and count["fv_fa"]<target["fv_fa"]:

        if n_fakes==1 and len(fake_periods)==1:

            start,end = fake_periods[0]

            if end-start>0.5:

                save_path=f"{SAVE_ROOT}/fv_fa_{count['fv_fa']}.mp4"

                cut_video(path,start,end-start,save_path)

                base64_video=encode_video(save_path)

                response=client.chat.completions.create(
                    model="gemini-3-pro-preview",
                    messages=[
                        {"role":"system","content":PROMPT_FV_FA},
                        {"role":"user","content":[
                            {"type":"image_url","image_url":{
                                "url":f"data:video/mp4;base64,{base64_video}"
                            }}
                        ]}
                    ]
                )

                output=response.choices[0].message.content

                entry={
                    "path":save_path,
                    "id":f"fv_fa_{count['fv_fa']}",
                    "label":"fake_video_fake_audio",
                    "question":QUESTION_FV_FA,
                    "answer":output
                }

                meta.append(entry)
                count["fv_fa"]+=1

    # ======================
    # fv_ra
    # ======================

    if modify_video and not modify_audio and count["fv_ra"]<target["fv_ra"]:

        if n_fakes==1 and len(fake_periods)==1:

            start,end=fake_periods[0]

            if end-start>0.5:

                save_path=f"{SAVE_ROOT}/fv_ra_{count['fv_ra']}.mp4"

                cut_video_visual_only(path,start,end-start,save_path)

                base64_video=encode_video(save_path)

                response=client.chat.completions.create(
                    model="gemini-3-pro-preview",
                    messages=[
                        {"role":"system","content":PROMPT_FV_RA},
                        {"role":"user","content":[
                            {"type":"image_url","image_url":{
                                "url":f"data:video/mp4;base64,{base64_video}"
                            }}
                        ]}
                    ]
                )

                output=response.choices[0].message.content

                entry={
                    "path":save_path,
                    "id":f"fv_ra_{count['fv_ra']}",
                    "label":"fake_video_real_audio",
                    "question":QUESTION_FV_RA,
                    "answer":output
                }

                meta.append(entry)
                count["fv_ra"]+=1

    # ======================
    # rv_fa
    # ======================

    if (not modify_video) and modify_audio and count["rv_fa"]<target["rv_fa"]:

        if n_fakes==1 and len(fake_periods)==1:

            start,end=fake_periods[0]

            if end-start>0.5:

                save_path=f"{SAVE_ROOT}/rv_fa_{count['rv_fa']}.wav"

                cut_audio(path,start,end-start,save_path)

                base64_audio=encode_audio(save_path)

                response=client.chat.completions.create(
                    model="gemini-3-pro-preview",
                    messages=[
                        {"role":"system","content":PROMPT_RV_FA},
                        {"role":"user","content":[
                            {"type":"input_audio","input_audio":{
                                "data":f"data:;base64,{base64_audio}",
                                "format":"wav"
                            }}
                        ]}
                    ]
                )

                output=response.choices[0].message.content

                entry={
                    "path":save_path,
                    "id":f"rv_fa_{count['rv_fa']}",
                    "label":"real_video_fake_audio",
                    "question":QUESTION_RV_FA,
                    "answer":output
                }

                meta.append(entry)
                count["rv_fa"]+=1

    # ======================
    # rv_ra
    # ======================

    if (not modify_video) and (not modify_audio) and count["rv_ra"]<target["rv_ra"]:

        save_path=f"{SAVE_ROOT}/rv_ra_{count['rv_ra']}.mp4"

        cut_video(path,0,1,save_path)

        base64_video=encode_video(save_path)

        response=client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[
                {"role":"system","content":PROMPT_RV_RA},
                {"role":"user","content":[
                    {"type":"image_url","image_url":{
                        "url":f"data:video/mp4;base64,{base64_video}"
                    }}
                ]}
            ]
        )

        output=response.choices[0].message.content

        entry={
            "path":save_path,
            "id":f"rv_ra_{count['rv_ra']}",
            "label":"real_video_real_audio",
            "question":QUESTION_RV_RA,
            "answer":output
        }

        meta.append(entry)
        count["rv_ra"]+=1


# =========================
# 保存json
# =========================

with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
    json.dump(meta,f,ensure_ascii=False,indent=4)

print("数据集生成完成")