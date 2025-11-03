#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from http import server
import os
import logging
import argparse
import re
import gradio as gr
from funasr import AutoModel
from videoclipper import VideoClipper
from llm.openai_api import openai_call
from llm.qwen_api import call_qwen_model
from llm.g4f_openai_api import g4f_openai_call
from utils.trans_utils import extract_timestamps
from introduction import top_md_1, top_md_3, top_md_4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--lang', '-l', type=str, default = "zh", help="language")
    parser.add_argument('--share', '-s', action='store_true', help="if to establish gradio share link")
    parser.add_argument('--port', '-p', type=int, default=7860, help='port number')
    parser.add_argument('--listen', action='store_true', help="if to listen to all hosts")
    args = parser.parse_args()
    
    if args.lang == 'zh':
        funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                disable_update=True
                                )
    else:
        funasr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                disable_update=True
                                )
    audio_clipper = VideoClipper(funasr_model)
    audio_clipper.lang = args.lang
    
    server_name='127.0.0.1'
    if args.listen:
        server_name = '0.0.0.0'
        

    def video_recog(video_input, sd_switch, hotwords, output_dir):
        """
        Call the audio_clipper.video_recog and convert the returned SRT/text
        into an array of non-empty lines. Return (res_text, lines_list, video_state).
        The caller (UI handlers) will decide how to display these values.
        """
        res = audio_clipper.video_recog(video_input, sd_switch, hotwords, output_dir=output_dir)
        # audio_clipper.video_recog is expected to return (res_text, res_srt, video_state)
        if not res:
            return None, [], None
        # Unpack defensively
        try:
            res_text, res_srt, video_state = res
        except Exception:
            # If the returned shape is unexpected, forward it through conservatively
            return res[0] if len(res) > 0 else None, [], res[2] if len(res) > 2 else None

        # Convert the SRT/subtitle string into a list of subtitle entries.
        # Many SRTs are formatted as blocks separated by blank lines like:
        # 1\n00:00:00,050 --> 00:00:02,010\nText line
        # To avoid returning a single-number line (the index) alone, we join
        # each block's non-empty lines into a single-line entry.
        if isinstance(res_srt, str):
            blocks = re.split(r"\n\s*\n", res_srt.strip()) if res_srt.strip() else []
            lines = []
            for block in blocks:
                parts = [ln.strip() for ln in block.splitlines() if ln.strip()]
                if not parts:
                    continue
                # Join the parts so each subtitle block becomes one line.
                # Example: ['1','00:00:00 --> 00:00:02','Some text'] -> '1 00:00:00 --> 00:00:02 Some text'
                entry = " ".join(parts)
                lines.append(entry)
        elif isinstance(res_srt, list):
            # If the underlying function already returned a list, use it.
            lines = res_srt
        else:
            lines = []

        return res_text, lines, video_state

    def video_clip(dest_text, video_spk_input, start_ost, end_ost, state, output_dir):
        return audio_clipper.video_clip(dest_text, start_ost, end_ost, state, dest_spk=video_spk_input, output_dir=output_dir)

    def mix_recog(video_input, hotwords, output_dir):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        video_state = None
        if video_input is not None:
            res_text, res_srt_lines, video_state = video_recog(video_input, 'No', hotwords, output_dir=output_dir)
            # Prepare the SRT string for the single SRT output box
            srt_string = "\n".join(res_srt_lines)
            # Prepare updates for the 100 pre-created checkboxes+textboxes
            check_updates = []
            text_updates = []
            max_lines = len(generated_textboxes)
            for i in range(max_lines):
                if i < len(res_srt_lines):
                    check_updates.append(gr.update(value=False, visible=True))
                    text_updates.append(gr.update(value=res_srt_lines[i], visible=True))
                else:
                    check_updates.append(gr.update(value=False, visible=False))
                    text_updates.append(gr.update(value="", visible=False))
            # Return values: video_text_output, video_srt_output, video_state, audio_state, <100 check updates>, <100 text updates>
            return [res_text, srt_string, video_state, None] + check_updates + text_updates

    def mix_recog_speaker(video_input, hotwords, output_dir):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        video_state = None
        if video_input is not None:
            res_text, res_srt_lines, video_state = video_recog(video_input, 'Yes', hotwords, output_dir=output_dir)
            srt_string = "\n".join(res_srt_lines)
            check_updates = []
            text_updates = []
            max_lines = len(generated_textboxes)
            for i in range(max_lines):
                if i < len(res_srt_lines):
                    check_updates.append(gr.update(value=False, visible=True))
                    text_updates.append(gr.update(value=res_srt_lines[i], visible=True))
                else:
                    check_updates.append(gr.update(value=False, visible=False))
                    text_updates.append(gr.update(value="", visible=False))
            return [res_text, srt_string, video_state, None] + check_updates + text_updates

    def mix_clip(dest_text, video_spk_input, start_ost, end_ost, video_state, output_dir):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(dest_text, start_ost, end_ost, video_state, dest_spk=video_spk_input, output_dir=output_dir)
            return clip_video_file, None, message, clip_srt

    def video_clip_addsub(dest_text, video_spk_input, start_ost, end_ost, state, output_dir, font_size, font_color):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        return audio_clipper.video_clip(dest_text, start_ost, end_ost, state, font_size=font_size, font_color=font_color, add_sub=True, dest_spk=video_spk_input, output_dir=output_dir)
        
    def llm_inference(system_content, user_content, srt_text, model, apikey):
        SUPPORT_LLM_PREFIX = ['qwen', 'gpt', 'g4f', 'moonshot', 'deepseek']
        if model.startswith('qwen'):
            return call_qwen_model(apikey, model, user_content+'\n'+srt_text, system_content)
        if model.startswith('gpt') or model.startswith('moonshot') or model.startswith('deepseek'):
            return openai_call(apikey, model, system_content, user_content+'\n'+srt_text)
        elif model.startswith('g4f'):
            model = "-".join(model.split('-')[1:])
            return g4f_openai_call(model, system_content, user_content+'\n'+srt_text)
        else:
            logging.error("LLM name error, only {} are supported as LLM name prefix.".format(SUPPORT_LLM_PREFIX))
    
    def AI_clip(LLM_res, dest_text, video_spk_input, start_ost, end_ost, video_state, output_dir):
        timestamp_list = extract_timestamps(LLM_res)
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(dest_text, start_ost, end_ost, video_state, dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=False)
            return clip_video_file, None, message, clip_srt

    def AI_clip_subti(LLM_res, dest_text, video_spk_input, start_ost, end_ost, video_state, output_dir):
        timestamp_list = extract_timestamps(LLM_res)
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(dest_text, start_ost, end_ost, video_state, dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=True)
            return clip_video_file, None, message, clip_srt


    # Function that populates pre-created checkboxes+textboxes (max 100 lines)
    def generate_textboxes():
        max_lines = 100
        index_updates = []
        check_updates = []
        text_updates = []
        # For each pre-created trio, set index/checkbox/textbox visibility and values
        for i in range(max_lines):
            index_updates.append(gr.update(visible=False))
            check_updates.append(gr.update(value=False, visible=False))
            text_updates.append(gr.update(value="", visible=False)) 

        # Return updates in the order: indexes, checks, then textboxes
        return index_updates + check_updates + text_updates


    def toggle_all_checks(toggle_state):
        """Toggle all visible checkboxes to the given state."""
        max_lines = 100  # Match the number of pre-created boxes
        updates = []
        for i in range(max_lines):
            # Set checked state but preserve visibility - only toggle visible ones
            updates.append(gr.update(value=toggle_state))
        return updates

    def on_textbox_focus(textbox_value, textbox_index, *checkbox_states):
        """When a textbox gets focus, check its checkbox if the textbox has content."""
        updates = list(checkbox_states)  # Keep existing states
        # If the textbox has content, ensure its checkbox is checked
        if textbox_value and textbox_value.strip():
            updates[textbox_index] = True
        return updates

    def collect_selected(*vals):
        """Collect checked subtitle texts and return them joined by '#' in id asc order."""
        if not vals:
            return ""
        half = len(vals) // 2
        checks = list(vals[:half])
        texts = list(vals[half:])
        selected = [texts[i] for i, c in enumerate(checks) if c and texts[i].strip()]
        # Clean each selected entry: remove any leading index and timestamps,
        # keep only the subtitle textual content. Return each selected entry on
        # its own line.
        cleaned = []
        ts_pattern = re.compile(r"\d{1,2}:\d{2}:\d{2}[,\.:]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.:]\d{1,3}")
        idx_pattern = re.compile(r"^\s*\d+\s*")
        for entry in selected:
            s = entry or ""
            # Remove leading index (e.g., '1 ')
            s = idx_pattern.sub('', s)
            # Remove timestamp ranges
            s = ts_pattern.sub('', s)
            # Remove any leftover leading/trailing punctuation and spaces
            s = s.strip(' -:;\t\n\r')
            s = s.strip()
            if s:
                cleaned.append(s)
        return "。".join(cleaned)



    # gradio interface
    theme = gr.Theme.load(os.path.join(os.path.dirname(__file__), "utils", "theme.json"))
    with gr.Blocks(theme=theme) as funclip_service:
        # tiny CSS to tighten layout, center checkboxes and make them bigger/darker
        gr.HTML("""
            <style>
            /* Make generated rows align items center and reduce gap */
            .generated-row { 
                display: flex !important; 
                align-items: center !important; 
                gap: 8px !important;
                padding: 4px !important;
                border-radius: 4px !important;
                transition: background-color 0.2s !important;
            }
            
            .generated-row:hover {
                background-color: rgba(0,0,0,0.05) !important;
            }

            /* Style the select all row */
            .select-all-row { 
                display: flex !important;
                align-items: center !important;
                padding: 12px !important;
                margin-bottom: 12px !important;
                background: #e9ecef !important;
                border-radius: 6px !important;
                border: 1px solid #ced4da !important;
            }

            /* Style the select all checkbox */
            .select-all-checkbox input[type="checkbox"] {
                transform: scale(1.5);
                accent-color: #0056b3;
                cursor: pointer;
                margin-right: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .select-all-checkbox label {
                font-weight: 600;
                color: #2c3e50;
            }

            /* Narrow checkbox column and center the checkbox inside it */
            .generated-row .checkbox-col { 
                display: flex;
                align-items: center;
                justify-content: center;
                width: 44px;
                padding: 0 4px;
            }

            /* Make checkboxes more visible with better contrast and bigger size */
            .generated-row .checkbox-col input[type="checkbox"] {
                transform: scale(1.5);
                accent-color: #0056b3;
                cursor: pointer;
                margin: 0 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 2px solid #0056b3;
                border-radius: 4px;
                position: relative;
                opacity: 1 !important;
            }

            .generated-row .checkbox-col input[type="checkbox"]:checked {
                background-color: #0056b3;
            }

            /* Give textbox column remaining space */
            .generated-row .textbox-col { 
                flex: 1 1 auto;
                cursor: text;
            }

            /* Enhanced textbox styling */
            .generated-row .textbox-col label, 
            .generated-row .textbox-col .gradio-textbox { 
                font-weight: 500;
                color: #2c3e50;
            }

            .generated-row .textbox-col .gradio-textbox:focus-within {
                border-color: #0056b3;
                box-shadow: 0 0 0 1px #0056b3;
            }
            </style>
        """)
        video_state, audio_state = gr.State(), gr.State()
        with gr.Row():
            video_input = gr.Video(label="视频输入 | Video Input")
        with gr.Row():
            gr.Examples(
                ['https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E5%A4%9A%E8%AF%BB%E4%B9%A6%EF%BC%9F%E8%BF%99%E6%98%AF%E6%88%91%E5%90%AC%E8%BF%87%E6%9C%80%E5%A5%BD%E7%9A%84%E7%AD%94%E6%A1%88-%E7%89%87%E6%AE%B5.mp4',
                    'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/2022%E4%BA%91%E6%A0%96%E5%A4%A7%E4%BC%9A_%E7%89%87%E6%AE%B52.mp4',
                    'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E4%BD%BF%E7%94%A8chatgpt_%E7%89%87%E6%AE%B5.mp4'],
                [video_input],
                label='示例视频 | Demo Video')
            gr.Examples(['https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E8%AE%BF%E8%B0%88.mp4'],
                        [video_input],
                        label='多说话人示例视频 | Multi-speaker Demo Video')
        with gr.Row():
            # with gr.Row():
            # video_sd_switch = gr.Radio(["No", "Yes"], label="👥区分说话人 Get Speakers", value='No')
            hotwords_input = gr.Textbox(label="🚒 热词 | Hotwords(可以为空，多个热词使用空格分隔，仅支持中文热词)")
            output_dir = gr.Textbox(label="📁 文件输出路径 | File Output Dir (可以为空，Linux, mac系统可以稳定使用)", value=" ")
        with gr.Row():
            recog_button = gr.Button("👂 识别 | ASR", variant="primary")
            recog_button2 = gr.Button("👂👫 识别+区分说话人 | ASR+SD")
        with gr.Row():    
            video_text_output = gr.Textbox(label="✏️ 识别结果 | Recognition Result", lines=10)
            video_srt_output = gr.Textbox(label="📖 SRT字幕内容 | RST Subtitles", lines=10)

        #with gr.Row():
            #generate_btn = gr.Button("Generate TextBoxes Dynamically")
        # Pre-create up to 100 hidden Textbox components. We'll update and show
        # the ones we need when the button is clicked. This avoids trying to
        # create new components after launch (which Gradio doesn't support).
        with gr.Row():
            with gr.Column():
                # Button to collect checked subtitles into `video_text_input`
                collect_selected_btn = gr.Button("Add Selected to Clip Text")
                # Add select/deselect all control
                with gr.Row(elem_classes="select-all-row"):
                    select_all = gr.Checkbox(label="Select All", value=False, elem_classes="select-all-checkbox")
                
                generated_checks = []
                generated_textboxes = []
                for i in range(100):
                    # Each generated row: small checkbox column + wide textbox column.
                    # Use elem_classes to allow CSS centering and tighter layout.
                    with gr.Row(elem_classes="generated-row"):
                        chk = gr.Checkbox(label="", value=False, visible=False, scale=1, elem_classes="checkbox-col")
                        tb = gr.Textbox(label=f"Line {i+1}", visible=False, scale=9, elem_classes="textbox-col", 
                                    elem_id=f"generated_textbox_{i}", interactive=True)
                        # Focus event will auto-select checkbox when textbox has content
                        tb.select(fn=lambda v, i=i: on_textbox_focus(v, i, *[c.value for c in generated_checks]),
                                inputs=[tb] + generated_checks,
                                outputs=generated_checks)
                    generated_checks.append(chk)
                    generated_textboxes.append(tb)
                


        with gr.Row():
            with gr.Column():
                with gr.Tab("🧠 LLM智能裁剪 | LLM Clipping"):
                    with gr.Column():
                        prompt_head = gr.Textbox(label="Prompt System (按需更改，最好不要变动主体和要求)", value=("你是一个视频srt字幕分析剪辑器，输入视频的srt字幕，"
                                "分析其中的精彩且尽可能连续的片段并裁剪出来，输出四条以内的片段，将片段中在时间上连续的多个句子及它们的时间戳合并为一条，"
                                "注意确保文字与时间戳的正确匹配。输出需严格按照如下格式：1. [开始时间-结束时间] 文本，注意其中的连接符是“-”"))
                        prompt_head2 = gr.Textbox(label="Prompt User（不需要修改，会自动拼接左下角的srt字幕）", value=("这是待裁剪的视频srt字幕："))
                        with gr.Column():
                            with gr.Row():
                                llm_model = gr.Dropdown(
                                    choices=[
                                        "deepseek-chat"
                                        "qwen-plus",
                                        "gpt-3.5-turbo", 
                                        "gpt-3.5-turbo-0125", 
                                        "gpt-4-turbo",
                                        "g4f-gpt-3.5-turbo"
                                    ], 
                                    value="deepseek-chat",
                                    label="LLM Model Name",
                                    allow_custom_value=True)
                                apikey_input = gr.Textbox(label="APIKEY")
                            llm_button =  gr.Button("LLM推理 | LLM Inference（首先进行识别，非g4f需配置对应apikey）", variant="primary")
                        llm_result = gr.Textbox(label="LLM Clipper Result")
                        with gr.Row():
                            llm_clip_button = gr.Button("🧠 LLM智能裁剪 | AI Clip", variant="primary")
                            llm_clip_subti_button = gr.Button("🧠 LLM智能裁剪+字幕 | AI Clip+Subtitles")
                with gr.Tab("✂️ 根据文本/说话人裁剪 | Text/Speaker Clipping"):
                    video_text_input = gr.Textbox(label="✏️ 待裁剪文本 | Text to Clip (多段文本使用'#'连接)", lines=10)
                    video_spk_input = gr.Textbox(label="✏️ 待裁剪说话人 | Speaker to Clip (多个说话人使用'#'连接)", lines=5)
                    with gr.Row():
                        clip_button = gr.Button("✂️ 裁剪 | Clip", variant="primary")
                        clip_subti_button = gr.Button("✂️ 裁剪+字幕 | Clip+Subtitles")
                    with gr.Row():
                        video_start_ost = gr.Slider(minimum=-500, maximum=1000, value=0, step=50, label="⏪ 开始位置偏移 | Start Offset (ms)")
                        video_end_ost = gr.Slider(minimum=-500, maximum=1000, value=100, step=50, label="⏩ 结束位置偏移 | End Offset (ms)")
                with gr.Row():
                    font_size = gr.Slider(minimum=10, maximum=100, value=32, step=2, label="🔠 字幕字体大小 | Subtitle Font Size")
                    font_color = gr.Radio(["black", "white", "green", "red"], label="🌈 字幕颜色 | Subtitle Color", value='white')
                    # font = gr.Radio(["黑体", "Alibaba Sans"], label="字体 Font")
                video_output = gr.Video(label="裁剪结果 | Video Clipped")
                clip_message = gr.Textbox(label="⚠️ 裁剪信息 | Clipping Log", lines=10)
                srt_clipped = gr.Textbox(label="📖 裁剪部分SRT字幕内容 | Clipped RST Subtitles", lines=10)
                
        # When recognition runs, also populate the pre-created generated_checks and generated_textboxes.
        recog_button.click(mix_recog, 
                    inputs=[video_input,hotwords_input,output_dir],
                    outputs=[video_text_output,video_srt_output,video_state,audio_state] + generated_checks + generated_textboxes)
        recog_button2.click(mix_recog_speaker, 
                    inputs=[video_input,hotwords_input,output_dir,],
                    outputs=[video_text_output,video_srt_output,video_state] + generated_checks + generated_textboxes)
        clip_button.click(mix_clip, 
                inputs=[video_text_input,video_spk_input,video_start_ost,video_end_ost,video_state,output_dir],
                outputs=[video_output,clip_message,srt_clipped])
        clip_subti_button.click(video_clip_addsub, 
                    inputs=[video_text_input,video_spk_input,video_start_ost,video_end_ost,video_state,output_dir,font_size,font_color],
                    outputs=[video_output, clip_message, srt_clipped])
        llm_button.click(llm_inference,
                inputs=[prompt_head, prompt_head2, video_srt_output, llm_model, apikey_input],
                outputs=[llm_result])
        llm_clip_button.click(AI_clip, 
                    inputs=[llm_result,video_text_input,video_spk_input,video_start_ost,video_end_ost,video_state,output_dir],
                    outputs=[video_output,clip_message,srt_clipped])
        llm_clip_subti_button.click(AI_clip_subti, 
                        inputs=[llm_result,video_text_input,video_spk_input,video_start_ost,video_end_ost,video_state,output_dir],
                        outputs=[video_output,clip_message,srt_clipped])

        # When button is clicked, populate the pre-created checks+textboxes.
        # generate_btn.click(fn=generate_textboxes, inputs=[], outputs=generated_checks + generated_textboxes)
        # Bind collect button to assemble checked lines into `video_text_input`.
        collect_selected_btn.click(collect_selected, inputs=generated_checks + generated_textboxes, outputs=[video_text_input])
        # Bind select all checkbox to toggle all visible checkboxes
        select_all.change(fn=toggle_all_checks, inputs=[select_all], outputs=generated_checks)
    
    # start gradio service in local or share
    if args.listen:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name, inbrowser=False)
    else:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name)
