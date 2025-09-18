import os
from moviepy.editor import VideoFileClip

def process_videos(directory):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为.mp4视频
        if filename.endswith(".mp4"):
            # 获取文件的完整路径
            video_file = os.path.join(directory, filename)
            # 加载视频
            clip = VideoFileClip(video_file)
            # 提取音频
            audio = clip.audio
            # 保存音频，文件名与原视频文件同名，但扩展名为.wav
            audio_output = os.path.join(directory, filename.replace(".mp4", ".wav"))
            audio.write_audiofile(audio_output)

# 使用方法
process_videos("C:\yk\mul_cremad\cremad\CREMA-D\Video")
