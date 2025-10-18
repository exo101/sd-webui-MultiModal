import os
import sys
import json
import time
from pathlib import Path

# 获取Index-TTS目录路径
index_tts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'index-tts')
model_dir = os.path.join(index_tts_path, "checkpoints")

# 添加路径到sys.path，确保正确导入
index_tts_dir = Path(index_tts_path).absolute()
sys_path_added = []

if str(index_tts_dir) not in sys.path:
    sys.path.insert(0, str(index_tts_dir))
    sys_path_added.append(str(index_tts_dir))
    
indextts_module_path = index_tts_dir / "indextts"
if str(indextts_module_path) not in sys.path:
    sys.path.insert(0, str(indextts_module_path))
    sys_path_added.append(str(indextts_module_path))
    
tools_module_path = index_tts_dir / "tools"
if str(tools_module_path) not in sys.path:
    sys.path.insert(0, str(tools_module_path))
    sys_path_added.append(str(tools_module_path))

# 延迟导入IndexTTS
tts = None
INDEX_TTS_AVAILABLE = True

def initialize_model():
    """
    初始化IndexTTS模型
    """
    global tts
    if tts is not None:
        return True
        
    try:
        # 确保路径已正确添加
        if index_tts_path not in sys.path:
            sys.path.insert(0, index_tts_path)
            
        if str(indextts_module_path) not in sys.path:
            sys.path.insert(0, str(indextts_module_path))
        
        # 尝试多种导入方式
        indextts = None
        try:
            import indextts
        except ImportError:
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("indextts", os.path.join(index_tts_path, "indextts", "__init__.py"))
                indextts = importlib.util.module_from_spec(spec)
                sys.modules["indextts"] = indextts
                spec.loader.exec_module(indextts)
            except Exception as e:
                # 最后尝试直接修改sys.path
                old_sys_path = sys.path[:]
                sys.path.insert(0, index_tts_path)
                try:
                    import indextts
                except Exception:
                    sys.path[:] = old_sys_path
                    raise
        
        from indextts.infer_v2 import IndexTTS2
        # 根据官方示范初始化模型，启用FP16半精度推理
        tts = IndexTTS2(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False
        )
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

def load_example_cases():
    """
    加载示例案例，参考原始项目实现
    """
    try:
        # 尝试加载i18n
        try:
            from tools.i18n.i18n import I18nAuto
            i18n = I18nAuto(language="Auto")
            EMO_CHOICES = [i18n("与音色参考音频相同"),
                           i18n("使用情感参考音频"),
                           i18n("使用情感向量控制"),
                           i18n("使用情感描述文本控制")]
        except Exception:
            # 如果i18n不可用，使用默认文本
            EMO_CHOICES = ["与音色参考音频相同",
                           "使用情感参考音频", 
                           "使用情感向量控制",
                           "使用情感描述文本控制"]
        
        # 返回空列表以删除所有示例案例
        example_cases = []
        return example_cases
    except Exception as e:
        print(f"加载示例案例时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def create_index_tts_ui():
    """
    创建IndexTTS UI界面
    """
    try:
        from tools.i18n.i18n import I18nAuto
        i18n = I18nAuto(language="Auto")
        
        # 支持的语言列表
        LANGUAGES = {
            "中文": "zh_CN",
            "English": "en_US"
        }
        
        EMO_CHOICES = [i18n("与音色参考音频相同"),
                       i18n("使用情感参考音频"),
                       i18n("使用情感向量控制"),
                       i18n("使用情感描述文本控制")]
        EMO_CHOICES_BASE = EMO_CHOICES[:3]
        EMO_CHOICES_EXPERIMENTAL = EMO_CHOICES
        
    except Exception as e:
        # 如果i18n不可用，使用默认文本
        EMO_CHOICES = ["与音色参考音频相同",
                       "使用情感参考音频", 
                       "使用情感向量控制",
                       "使用情感描述文本控制"]
        EMO_CHOICES_BASE = EMO_CHOICES[:3]
        EMO_CHOICES_EXPERIMENTAL = EMO_CHOICES

    import gradio as gr
    
    # 加载示例案例
    example_cases = load_example_cases()
    
    with gr.Blocks() as index_tts_ui:
        gr.HTML('''
        <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
        <p align="center">
        <a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
        </p>
        ''')
        
        with gr.Tab("音频生成"):
            with gr.Row():
                with gr.Column():
                    prompt_audio = gr.Audio(
                        label="音色参考音频",
                        sources=["upload"],
                        type="filepath"
                    )
                    
                    # 添加预设音频组件
                    with gr.Accordion("预设音色参考音频", open=True):
                        preset_audio_dir = os.path.join(index_tts_path, "预设")
                        if os.path.exists(preset_audio_dir):
                            preset_files = [f for f in os.listdir(preset_audio_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
                            preset_audio_paths = [os.path.join(preset_audio_dir, f) for f in preset_files]
                            
                            if preset_audio_paths:
                                gr.Examples(
                                    examples=preset_audio_paths,
                                    inputs=prompt_audio,
                                    label="点击或拖拽以下音频作为音色参考:",
                                    examples_per_page=len(preset_audio_paths)  # 显示所有音频文件
                                )
                            else:
                                gr.Markdown("暂无预设音频文件")
                        else:
                            gr.Markdown("预设音频文件夹不存在")
                    
                    input_text_single = gr.TextArea(
                        label="文本",
                        placeholder="请输入目标文本",
                        info="当前模型版本"
                    )
                    
                    # 添加示例文本
                    sample_texts = [
                        "你知道我这么多年，是怎么过来的，每天以泪洗面，痛不欲生！",
                        "本座乃是乱星海第一美人，谁敢与我争锋，还有谁敢不服！",
                        "本座在这天南大陆与乱星海闯荡这么多年，经历了多少生死攸关，就不能好好享受享受吗？接着奏乐接着舞！",
                        "哥哥，哥哥，你骑着小电动车带着我，你女朋友知道了不会吃醋吧？你女朋友知道我们吃一颗棒棒糖不会揍我吧？不像我，我只会心疼哥哥！"
                    ]
                    
                    with gr.Accordion("示例文本", open=False):
                        gr.Examples(
                            examples=sample_texts,
                            inputs=input_text_single,
                            label="点击以下示例文本可直接使用:",
                            examples_per_page=20
                        )
                    
                    gen_button = gr.Button("生成语音", variant="primary")
                    
                with gr.Column():
                    output_audio = gr.Audio(
                        label="生成结果",
                        type="filepath"
                    )
            
            experimental_checkbox = gr.Checkbox(label="显示实验功能", value=False)
            
            with gr.Accordion("功能设置"):
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_BASE,
                    value=EMO_CHOICES_BASE[0],
                    label="情感控制方式",
                    type="index"  # 使用索引类型
                )
            
            with gr.Group(visible=False) as emotion_reference_group:
                with gr.Row():
                    emo_upload = gr.Audio(label="上传情感参考音频", type="filepath")

            with gr.Row(visible=False) as emotion_randomize_group:
                emo_random = gr.Checkbox(label="情感随机采样", value=False)

            with gr.Group(visible=False) as emotion_vector_group:
                with gr.Row():
                    with gr.Column():
                        vec1 = gr.Slider(label="喜", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec2 = gr.Slider(label="怒", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec3 = gr.Slider(label="哀", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec4 = gr.Slider(label="惧", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    with gr.Column():
                        vec5 = gr.Slider(label="厌恶", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec6 = gr.Slider(label="低落", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec7 = gr.Slider(label="惊喜", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec8 = gr.Slider(label="平静", minimum=0.0, maximum=1.0, value=0.0, step=0.05)

            with gr.Group(visible=False) as emo_text_group:
                with gr.Row():
                    emo_text = gr.Textbox(
                        label="情感描述文本",
                        placeholder="请输入情绪描述（或留空以自动使用目标文本作为情绪描述）",
                        value="",
                        info="例如：委屈巴巴、危险在悄悄逼近"
                    )

            with gr.Row(visible=False) as emo_weight_group:
                emo_weight = gr.Slider(label="情感权重", minimum=0.0, maximum=1.0, value=0.8, step=0.01)

            with gr.Accordion("高级生成参数设置", open=False, visible=False) as advanced_settings_group:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**GPT2 采样设置** _参数会影响音频多样性和生成速度详见 [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                        with gr.Row():
                            do_sample = gr.Checkbox(label="do_sample", value=True, info="是否进行采样")
                            temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                        with gr.Row():
                            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                            top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                            num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                        with gr.Row():
                            repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                            length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                        max_mel_tokens = gr.Slider(
                            label="max_mel_tokens", 
                            value=1500, 
                            minimum=50, 
                            maximum=2000, 
                            step=10, 
                            info="生成Token最大数量，过小导致音频被截断"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown('**分句设置** _参数会影响音频质量和生成速度_')
                        with gr.Row():
                            max_text_tokens_per_segment = gr.Slider(
                                label="分句最大Token数", 
                                value=120, 
                                minimum=20, 
                                maximum=400, 
                                step=2,
                                info="建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"
                            )
                        with gr.Accordion("预览分句结果", open=True) as segments_settings:
                            segments_preview = gr.Dataframe(
                                headers=["序号", "分句内容", "Token数"],
                                wrap=True,
                            )
                
                advanced_params = [
                    do_sample, top_p, top_k, temperature,
                    length_penalty, num_beams, repetition_penalty, max_mel_tokens
                ]
        
        def process_audio(emo_control_method, prompt, text,
                          emo_ref_path, emo_weight,
                          vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                          emo_text, emo_random,
                          max_text_tokens_per_segment,
                          do_sample, top_p, top_k, temperature,
                          length_penalty, num_beams, repetition_penalty, max_mel_tokens):
            try:
                # 初始化模型
                if not initialize_model():
                    return gr.update(value=None)
                
                global tts
                
                # 检查必要参数
                if not prompt or not text:
                    print("缺少必要参数：参考音频或文本")
                    return gr.update(value=None)
                
                # 准备情感向量
                emo_vector = None
                if emo_control_method == 2:  # emotion vectors
                    emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
                    # 情感因子，提供更好的用户体验
                    k_vec = [0.75, 0.70, 0.80, 0.80, 0.75, 0.75, 0.55, 0.45]
                    import numpy as np
                    tmp = np.array(k_vec) * np.array(emo_vector)
                    if np.sum(tmp) > 0.8:
                        tmp = tmp * 0.8 / np.sum(tmp)
                    emo_vector = tmp.tolist()
                
                # 准备情感文本
                use_emo_text = emo_control_method == 3  # emotion text description
                
                # 设置推理参数
                generation_kwargs = {
                    "do_sample": bool(do_sample),
                    "top_p": float(top_p),
                    "top_k": int(top_k) if int(top_k) > 0 else None,
                    "temperature": float(temperature),
                    "length_penalty": float(length_penalty),
                    "num_beams": int(num_beams),
                    "repetition_penalty": float(repetition_penalty),
                    "max_mel_tokens": int(max_mel_tokens),
                    "max_text_tokens_per_segment": int(max_text_tokens_per_segment)
                }
                
                # 生成音频文件路径
                output_path = os.path.join(index_tts_path, "temp", f"output_{int(time.time())}.wav")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 调用推理函数
                result = tts.infer(
                    spk_audio_prompt=prompt,
                    text=text,
                    output_path=output_path,
                    emo_audio_prompt=emo_ref_path if emo_control_method == 1 else None,
                    emo_alpha=emo_weight * 0.8 if emo_control_method == 1 else emo_weight,
                    emo_vector=emo_vector,
                    use_emo_text=use_emo_text,
                    emo_text=emo_text if use_emo_text else None,
                    use_random=emo_random if emo_control_method in [2, 3] else False,
                    verbose=True,
                    **generation_kwargs
                )
                
                # 返回生成的音频文件
                return gr.update(value=output_path)
                
            except Exception as e:
                print(f"处理音频时出错: {e}")
                import traceback
                traceback.print_exc()
                return gr.update(value=None)
        
        def on_method_select(emo_control_method):
            # 使用索引值进行判断
            if emo_control_method == 1:  # emotion reference audio
                return (
                    gr.update(visible=True),   # emotion_reference_group
                    gr.update(visible=True),   # emotion_randomize_group
                    gr.update(visible=False),  # emotion_vector_group
                    gr.update(visible=False),  # emo_text_group
                    gr.update(visible=True)    # emo_weight_group
                )
            elif emo_control_method == 2:  # emotion vectors
                return (
                    gr.update(visible=False),  # emotion_reference_group
                    gr.update(visible=True),   # emotion_randomize_group
                    gr.update(visible=True),   # emotion_vector_group
                    gr.update(visible=False),  # emo_text_group
                    gr.update(visible=False)   # emo_weight_group
                )
            elif emo_control_method == 3:  # emotion text description
                return (
                    gr.update(visible=False),  # emotion_reference_group
                    gr.update(visible=True),   # emotion_randomize_group
                    gr.update(visible=False),  # emotion_vector_group
                    gr.update(visible=True),   # emo_text_group
                    gr.update(visible=True)    # emo_weight_group
                )
            else:  # 0: same as speaker voice
                return (
                    gr.update(visible=False),  # emotion_reference_group
                    gr.update(visible=False),  # emotion_randomize_group
                    gr.update(visible=False),  # emotion_vector_group
                    gr.update(visible=False),  # emo_text_group
                    gr.update(visible=False)   # emo_weight_group
                )
        
        def on_experimental_change(is_exp):
            if is_exp:
                return gr.update(choices=EMO_CHOICES_EXPERIMENTAL, value=EMO_CHOICES_EXPERIMENTAL[0]), gr.update(visible=True)
            else:
                return gr.update(choices=EMO_CHOICES_BASE, value=EMO_CHOICES_BASE[0]), gr.update(visible=False)
        
        # 绑定事件
        emo_control_method.select(
            fn=on_method_select,
            inputs=[emo_control_method],
            outputs=[
                emotion_reference_group, 
                emotion_randomize_group, 
                emotion_vector_group, 
                emo_text_group, 
                emo_weight_group
            ]
        )
        
        experimental_checkbox.change(
            fn=on_experimental_change,
            inputs=[experimental_checkbox],
            outputs=[emo_control_method, advanced_settings_group]
        )
        
        gen_button.click(
            fn=process_audio,
            inputs=[emo_control_method, prompt_audio, input_text_single, 
                   emo_upload, emo_weight,
                   vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                   emo_text, emo_random,
                   max_text_tokens_per_segment,
                   *advanced_params],
            outputs=[output_audio]
        )
    
    return {
        "ui": index_tts_ui,
        "prompt_audio": prompt_audio,
        "input_text_single": input_text_single,
        "gen_button": gen_button,
        "output_audio": output_audio,
        "experimental_checkbox": experimental_checkbox,
        "emo_control_method": emo_control_method,
        "emotion_reference_group": emotion_reference_group,
        "emotion_randomize_group": emotion_randomize_group,
        "emotion_vector_group": emotion_vector_group,
        "emo_text_group": emo_text_group,
        "emo_weight_group": emo_weight_group,
        "advanced_settings_group": advanced_settings_group,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "max_mel_tokens": max_mel_tokens,
        "max_text_tokens_per_segment": max_text_tokens_per_segment,
        "segments_preview": segments_preview,
        "emo_upload": emo_upload,
        "emo_random": emo_random,
        "vec1": vec1,
        "vec2": vec2,
        "vec3": vec3,
        "vec4": vec4,
        "vec5": vec5,
        "vec6": vec6,
        "vec7": vec7,
        "vec8": vec8,
        "emo_text": emo_text,
        "emo_weight": emo_weight
    }

__all__ = ['create_index_tts_ui', 'INDEX_TTS_AVAILABLE']