#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from http import server
import os
import logging
import argparse
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
        return audio_clipper.video_recog(video_input, sd_switch, hotwords, output_dir=output_dir)

    def mix_recog(video_input, hotwords, output_dir):
        output_dir = output_dir.strip()
        if not len(output_dir):
            output_dir = None
        else:
            output_dir = os.path.abspath(output_dir)
        video_state = None
        if video_input is not None:
            res_text, res_srt, video_state = video_recog(video_input, 'No', hotwords, output_dir=output_dir)
            return res_text, res_srt, video_state, None

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

    # gradio interface
    theme = gr.Theme.load(os.path.join(os.path.dirname(__file__), "utils", "theme.json"))
    with gr.Blocks(theme=theme) as funclip_service:
        video_state, audio_state = gr.State(), gr.State()
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    video_input = gr.Video(label="视频输入 | Video Input")
                with gr.Column():
                    # gr.Examples(['https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E5%A4%9A%E8%AF%BB%E4%B9%A6%EF%BC%9F%E8%BF%99%E6%98%AF%E6%88%91%E5%90%AC%E8%BF%87%E6%9C%80%E5%A5%BD%E7%9A%84%E7%AD%94%E6%A1%88-%E7%89%87%E6%AE%B5.mp4', 
                    #              'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/2022%E4%BA%91%E6%A0%96%E5%A4%A7%E4%BC%9A_%E7%89%87%E6%AE%B52.mp4', 
                    #              'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E4%BD%BF%E7%94%A8chatgpt_%E7%89%87%E6%AE%B5.mp4'],
                    #             [video_input],
                    #             label='示例视频 | Demo Video')
                    
                    gr.Examples(['https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ClipVideo/%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E5%A4%9A%E8%AF%BB%E4%B9%A6%EF%BC%9F%E8%BF%99%E6%98%AF%E6%88%91%E5%90%AC%E8%BF%87%E6%9C%80%E5%A5%BD%E7%9A%84%E7%AD%94%E6%A1%88-%E7%89%87%E6%AE%B5.mp4'],
                                [video_input],
                                label='示例视频 | Demo Video')
                    with gr.Column():
                        hotwords_input = gr.Textbox(label="🚒 热词 | Hotwords(可以为空，多个热词使用空格分隔，仅支持中文热词)", visible=False)
                        output_dir = gr.Textbox(label="📁 文件输出路径 | File Output Dir (可以为空，Linux, mac系统可以稳定使用)", value=" ", visible=False)
                        with gr.Row():
                            recog_button = gr.Button("👂 识别 | ASR", variant="primary")
                video_text_output = gr.Textbox(label="✏️ 识别结果 | Recognition Result", lines=10)
                video_srt_output = gr.Textbox(label="📖 SRT字幕内容 | RST Subtitles", lines=10)
            with gr.Column():
                with gr.Column("✂️ 根据文本/说话人裁剪 | Text/Speaker Clipping"):
                    video_text_input = gr.Textbox(label="✏️ 待裁剪文本 | Text to Clip (多段文本使用'#'连接)", lines=3)
                    video_spk_input = gr.Textbox(label="✏️ 待裁剪说话人 | Speaker to Clip (多个说话人使用'#'连接)", visible=False)
                    with gr.Row():
                        clip_button = gr.Button("✂️ 裁剪 | Clip", variant="primary")
                        clip_subti_button = gr.Button("✂️ 裁剪+字幕 | Clip+Subtitles", variant="huggingface")
                    with gr.Column():
                        video_start_ost = gr.Slider(minimum=-500, maximum=1000, value=0, step=50, label="⏪ 开始位置偏移 | Start Offset (ms)")
                        video_end_ost = gr.Slider(minimum=-500, maximum=1000, value=100, step=50, label="⏩ 结束位置偏移 | End Offset (ms)")
                
                with gr.Row():
                    font_size = gr.Slider(minimum=10, maximum=100, value=32, step=2, label="🔠 字幕字体大小 | Subtitle Font Size")
                    font_color = gr.Radio(["black", "white", "green", "red"], label="🌈 字幕颜色 | Subtitle Color", value='white')
                video_output = gr.Video(label="裁剪结果 | Video Clipped")
                srt_clipped = gr.Textbox(label="📖 裁剪部分SRT字幕内容 | Clipped RST Subtitles", lines=10)
                clip_message = gr.Textbox(label="⚠️ 裁剪信息 | Clipping Log", lines=2)
                
        recog_button.click(mix_recog, 
                            inputs=[video_input,hotwords_input,output_dir],
                            outputs=[video_text_output,video_srt_output,video_state,audio_state])
        clip_button.click(mix_clip, 
                            inputs=[video_text_input,video_spk_input,video_start_ost,video_end_ost,video_state,output_dir],
                            outputs=[video_output,clip_message,srt_clipped])
        clip_subti_button.click(video_clip_addsub, 
                            inputs=[video_text_input,video_spk_input,video_start_ost,video_end_ost,video_state,output_dir,font_size,font_color],
                            outputs=[video_output, clip_message, srt_clipped])
    
    # start gradio service in local or share
    if args.listen:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name, inbrowser=False)
    else:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name)
