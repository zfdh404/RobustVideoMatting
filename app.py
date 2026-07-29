import os
import re
import subprocess
import asyncio
import gradio as gr
import socket
import gc
import torch
import time

# 设置环境变量缓解显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Windows asyncio兼容
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==========================
# 配置
# ==========================
PYTHON_EXE = r"D:\ProgramFiles\miniconda3\envs\rvm\python.exe"

MODEL_PATH = {
    "ResNet50 高质量": {
        "variant": "resnet50",
        "checkpoint": "checkpoints/rvm_resnet50.pth"
    },
    "MobileNetV3 快速": {
        "variant": "mobilenetv3",
        "checkpoint": "checkpoints/rvm_mobilenetv3.pth"
    }
}

current_process = None
current_pid = None

# ==========================
# 辅助函数
# ==========================
def check_ffprobe():
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def get_video_dimensions(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError("视频文件不存在: " + video_path)
    if not check_ffprobe():
        raise RuntimeError("ffprobe 未找到或不可用！请安装 ffmpeg 并添加到 PATH。")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        if not output:
            raise RuntimeError("ffprobe 未返回尺寸信息，可能视频无视频流或已损坏。")
        nums = re.findall(r'\d+', output)
        if len(nums) < 2:
            raise RuntimeError("无法解析宽高: " + output)
        return int(nums[0]), int(nums[1])
    except Exception as e:
        raise RuntimeError("获取视频尺寸失败: " + str(e))

def smooth_alpha(alpha_path, output_path, blur_radius=1.5):
    cmd = [
        "ffmpeg", "-y",
        "-i", alpha_path,
        "-vf", "gblur=sigma={}".format(blur_radius),
        "-c:v", "libx264",
        "-pix_fmt", "gray",
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("平滑alpha失败: " + result.stderr)
    return output_path

# ==========================
# 生成透明MOV（支持填充）
# ==========================
def create_alpha_mov(foreground, alpha, output, target_w, target_h):
    """
    将 foreground 和 alpha 合成为带透明通道的 ProRes MOV，
    并居中放置在 target_w x target_h 的画布上。
    如果 foreground 尺寸与 target 相同，则直接合成（pad 无效果）。
    """
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", foreground]
        dim_out = subprocess.check_output(cmd, text=True).strip()
        nums = re.findall(r'\d+', dim_out)
        if len(nums) >= 2:
            fg_w, fg_h = int(nums[0]), int(nums[1])
        else:
            fg_w, fg_h = 0, 0
    except:
        fg_w, fg_h = 0, 0

    if fg_w != target_w or fg_h != target_h:
        pad_filter = "pad={}:{}:(ow-iw)/2:(oh-ih)/2:color=black@0".format(target_w, target_h)
        filter_complex = "[0:v]{}[fg];[1:v]{}[a];[fg][a]alphamerge".format(pad_filter, pad_filter)
    else:
        filter_complex = "[0:v][1:v]alphamerge"

    cmd = [
        "ffmpeg", "-y",
        "-i", foreground,
        "-i", alpha,
        "-filter_complex", filter_complex,
        "-c:v", "prores_ks",
        "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le",
        "-map_metadata", "-1",
        "-metadata:s:v:0", "rotate=0",
        output
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
    return result.stdout

# ==========================
# 停止任务
# ==========================
def stop_task():
    global current_process, current_pid
    msg = ""
    try:
        if current_pid:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(current_pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            msg += "已结束RVM进程\n"
        current_process = None
        current_pid = None
        return msg + "完成"
    except Exception as e:
        return str(e)

# ==========================
# 主处理
# ==========================
def run_rvm(
    video_file,
    model,
    downsample,
    seq_chunk,
    target_width,
    target_height,
    alpha_smooth,
    progress=gr.Progress()
):
    global current_process, current_pid

    if current_process is not None:
        return (None, None, "已有任务运行，请先结束任务")

    if video_file is None:
        return (None, None, "请选择视频")

    raw_input_video = video_file
    filename = os.path.basename(raw_input_video)
    name = os.path.splitext(filename)[0]

    workdir = os.path.join("output", name)
    os.makedirs(workdir, exist_ok=True)

    log = ""

    # ---------- 预处理：旋转校正 iPhone 视频 ----------
    progress(0.00, desc="正在预处理视频旋转方向...")
    input_rotated = os.path.join(workdir, "input_rotated.mp4")
    cmd_rotate = [
        "ffmpeg", "-y",
        "-i", raw_input_video,
        "-map", "0:v:0",                  # 只提取第一条视频轨，忽略 Apple 特有的 mebx 数据流
        "-vf", "format=yuv420p",          # 强制指定像素格式，FFmpeg 会自动应用 rotation 将物理像素转置为竖屏
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "ultrafast",
        "-metadata:s:v:0", "rotate=0",    # 抹除原有的 rotate 标记，防止后续处理二次旋转
        input_rotated
    ]
    try:
        subprocess.run(cmd_rotate, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return (None, None, "视频方向校正失败: " + e.stderr.decode(errors="ignore"))

    input_video = input_rotated

    # ---------- 获取物理校正后的真实尺寸 ----------
    try:
        ori_w, ori_h = get_video_dimensions(input_video)
    except Exception as e:
        return (None, None, "获取视频尺寸失败: " + str(e))
    log += f"\n物理解析尺寸: {ori_w}x{ori_h}"
    progress(0.01, desc="物理尺寸 {}x{}".format(ori_w, ori_h))

    # ---------- 计算缩放与最终输出画布尺寸 ----------
    if target_width > 0 and target_height > 0:
        out_w, out_h = target_width, target_height
        ratio_w = out_w / ori_w
        ratio_h = out_h / ori_h
        scale_ratio = min(ratio_w, ratio_h)
        new_w = int(ori_w * scale_ratio)
        new_h = int(ori_h * scale_ratio)
    elif target_width > 0:
        out_w = target_width
        scale_ratio = out_w / ori_w
        new_h = int(ori_h * scale_ratio)
        new_w = out_w
        out_h = new_h
    elif target_height > 0:
        out_h = target_height
        scale_ratio = out_h / ori_h
        new_w = int(ori_w * scale_ratio)
        new_h = out_h
        out_w = new_w
    else:
        out_w, out_h = ori_w, ori_h
        new_w, new_h = ori_w, ori_h
        scale_ratio = 1.0

    scale_ratio = max(0.1, min(scale_ratio, 10.0))
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1

    if target_width > 0 and target_height <= 0:
        out_w = new_w
        out_h = new_h
    elif target_height > 0 and target_width <= 0:
        out_w = new_w
        out_h = new_h

    log += f"\n目标输出画布: {out_w}x{out_h}"
    log += f"\n缩放后尺寸: {new_w}x{new_h}"
    log += f"\n缩放比例: {scale_ratio:.3f}"
    progress(0.03, desc="缩放至 {}x{}".format(new_w, new_h))

    # ---------- 缩放视频 ----------
    scaled_video = os.path.join(workdir, "input_scaled.mov")
    scale_filter = "scale={}:{}".format(new_w, new_h)
    cmd_scale = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", scale_filter,
        "-c:v", "prores_ks",
        "-profile:v", "4444",
        "-pix_fmt", "yuv444p10le",
        scaled_video
    ]
    try:
        subprocess.run(cmd_scale, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return (None, None, "缩放视频失败: " + e.stderr.decode(errors="ignore"))

    input_for_rvm = scaled_video
    cfg = MODEL_PATH[model]

    foreground_mp4 = os.path.join(workdir, "foreground.mp4")
    alpha_mp4 = os.path.join(workdir, "alpha.mp4")
    composition_mp4 = os.path.join(workdir, "composition.mp4")
    transparent = os.path.join("output", "{}_filled_{}x{}_transparent.mov".format(name, out_w, out_h))

    # ---------- RVM推理 ----------
    cmd = [
        PYTHON_EXE,
        "inference.py",
        "--variant", cfg["variant"],
        "--checkpoint", cfg["checkpoint"],
        "--device", "cuda",
        "--input-source", input_for_rvm,
        "--output-type", "video",
        "--output-composition", composition_mp4,
        "--output-alpha", alpha_mp4,
        "--output-foreground", foreground_mp4,
        "--seq-chunk", str(seq_chunk),
        "--downsample-ratio", str(downsample)
    ]

    try:
        progress(0.05, desc="启动RVM")
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        current_pid = current_process.pid
        for line in current_process.stdout:
            log += line
            print(line, end="")
            m = re.search(r"(\d+)%", line)
            if m:
                progress(int(m.group(1)) / 100, desc="处理中")
        current_process.wait()
    except Exception as e:
        return (None, None, "RVM运行异常: " + str(e))
    finally:
        current_process = None
        current_pid = None

    if not os.path.exists(foreground_mp4):
        return (None, None, log + "\n错误: 未生成 foreground.mp4")
    if not os.path.exists(composition_mp4):
        return (None, None, log + "\n错误: 未生成 composition.mp4")

    # 清理显存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log += "\n已清理显存缓存"

    # alpha平滑
    progress(0.8, desc="后处理alpha（平滑）")
    if alpha_smooth > 0:
        alpha_smoothed = os.path.join(workdir, "alpha_smoothed.mp4")
        try:
            smooth_alpha(alpha_mp4, alpha_smoothed, alpha_smooth)
            alpha_for_synth = alpha_smoothed
        except Exception as e:
            return (None, None, "平滑alpha失败: " + str(e))
    else:
        alpha_for_synth = alpha_mp4

    # 转码ProRes
    progress(0.85, desc="转码为ProRes")
    foreground_prores = os.path.join(workdir, "foreground_prores.mov")
    alpha_prores = os.path.join(workdir, "alpha_prores.mov")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", foreground_mp4,
            "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuv444p10le",
            foreground_prores
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([
            "ffmpeg", "-y", "-i", alpha_for_synth,
            "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuv444p10le",
            alpha_prores
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return (None, None, "转码ProRes失败: " + e.stderr.decode(errors="ignore"))

    progress(0.9, desc="生成透明MOV")
    ffmpeg_log = create_alpha_mov(foreground_prores, alpha_prores, transparent, out_w, out_h)
    log += ffmpeg_log

    time.sleep(0.5)

    # 清理临时文件
    for f in [input_rotated, scaled_video, foreground_mp4, alpha_mp4,
              alpha_for_synth, foreground_prores, alpha_prores]:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

    if not os.path.exists(transparent):
        return (None, None, log + "\n错误: 未生成透明MOV")
    if not os.path.exists(composition_mp4):
        return (None, None, log + "\n错误: composition.mp4 丢失")

    try:
        out_w2, out_h2 = get_video_dimensions(transparent)
        log += f"\n最终输出尺寸: {out_w2}x{out_h2}"
    except Exception as e:
        log += f"\n无法获取最终尺寸: {e}"

    progress(1, desc="完成")
    return (transparent, composition_mp4, log)

# ==========================
# 自动找端口
# ==========================
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# ==========================
# Gradio界面
# ==========================
with gr.Blocks(title="RVM透明视频工具") as demo:
    gr.Markdown("""
# Robust Video Matting（简化版）
**自动按比例缩放，并强制填充到目标画布（透明背景），保证内容不变形。**

- 只填宽度 → 高度自动按比例调整
- 只填高度 → 宽度自动按比例调整
- 两者都填 → 按最小比例缩放并居中，其余透明

**显存优化建议**：若显存不足，请降低目标分辨率、使用 MobileNetV3 模型、减小 seq_chunk。
""")

    video_input = gr.File(
        label="选择视频",
        file_types=[".mp4", ".mov", ".avi"],
        type="filepath"
    )

    model = gr.Dropdown(
        choices=list(MODEL_PATH.keys()),
        value="MobileNetV3 快速",
        label="模型"
    )

    with gr.Row():
        downsample = gr.Dropdown(
            choices=[0.125, 0.25, 0.5, 0.75, 1.0],
            value=0.25,
            label="Downsample Ratio"
        )
        seq_chunk = gr.Dropdown(
            choices=[8, 16, 32, 64],
            value=16,
            label="Seq Chunk"
        )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 缩放目标尺寸（像素）")
            with gr.Row():
                target_width = gr.Number(
                    value=1080,
                    label="目标宽度（填0自动）",
                    precision=0,
                    minimum=0,
                    step=1
                )
                target_height = gr.Number(
                    value=1920,
                    label="目标高度（填0自动）",
                    precision=0,
                    minimum=0,
                    step=1
                )
            gr.Markdown("*提示：只填宽度则高度按比例缩放；只填高度则宽度按比例缩放；两者都填则按最小比例缩放并填充（始终填充）。*")
        with gr.Column(scale=1):
            gr.Markdown("### 快速预设")
            preset = gr.Dropdown(
                choices=[
                    ("抖音竖屏 (1080x1920)", "1080,1920"),
                    ("原始", "0,0"),
                    ("1080p横屏 (1920x1080)", "1920,1080"),
                    ("720p横屏 (1280x720)", "1280,720"),
                    ("540p横屏 (960x540)", "960,540")
                ],
                value="1080,1920",
                label="选择预设"
            )
            def apply_preset(preset_val):
                w, h = map(int, preset_val.split(','))
                return {target_width: w, target_height: h}
            preset.change(apply_preset, inputs=preset, outputs=[target_width, target_height])

    alpha_smooth = gr.Slider(
        minimum=0,
        maximum=5.0,
        step=0.1,
        value=1.5,
        label="Alpha边缘平滑半径（高斯模糊）",
        info="值越大边缘越柔和，可减少锯齿，但可能损失细节。0表示不平滑。"
    )

    with gr.Row():
        start = gr.Button("开始处理", variant="primary")
        stop = gr.Button("结束任务", variant="stop")

    output_mov = gr.File(label="透明MOV（已填充）")
    preview_mp4 = gr.File(label="预览Composition MP4")
    log = gr.Textbox(label="日志", lines=30)

    start.click(
        fn=lambda: gr.update(interactive=False),
        outputs=start
    ).then(
        fn=run_rvm,
        inputs=[video_input, model, downsample, seq_chunk, target_width, target_height, alpha_smooth],
        outputs=[output_mov, preview_mp4, log]
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=start
    )

    stop.click(stop_task, outputs=log)

# ==========================
# 启动
# ==========================
port = find_free_port()
print(f"使用空闲端口: {port}")
demo.launch(
    server_name="127.0.0.1",
    server_port=port,
    inbrowser=True
)