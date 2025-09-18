import os
import glob
import subprocess
import shutil


def convert_flv_to_mp4(input_directory, output_directory):
    # 获取指定目录下的所有 flv 文件
    flv_files = glob.glob(os.path.join(input_directory, "*.flv"))

    for flv_file_path in flv_files:
        # 创建同名的 mp4 文件路径
        base_name = os.path.splitext(os.path.basename(flv_file_path))[0]
        mp4_file_path = os.path.join(output_directory, base_name + ".mp4")

        # 使用 ffmpeg 将 flv 文件转换为 mp4 文件
        command = f"ffmpeg -i {flv_file_path} {mp4_file_path}"
        subprocess.run(command, shell=True)


# 使用示例
convert_flv_to_mp4("C:\yk\mul_cremad\cremad\CREMA-D\VideoFlash", "C:\yk\mul_cremad\cremad\CREMA-D\Video")
