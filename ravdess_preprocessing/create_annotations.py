import os
import numpy as np
from collections import defaultdict

root = r'E:\雷导研究生\实验代码修改\小论文返稿\mul_cremad\cremad\CREMA-D\Video'


# 获取所有演员ID
def get_speaker_ids():
    speaker_ids = set()
    files = [f for f in os.listdir(root) if f.endswith('.npy') and 'croppad' in f]
    for file in files:
        speaker_id = file.split('_')[0]  # 假设文件名格式为 "1001_XXX_XXX.npy"
        speaker_ids.add(speaker_id)
    return sorted(list(speaker_ids))


# 按说话人分组文件
def group_files_by_speaker():
    speaker_files = defaultdict(list)
    files = [f for f in os.listdir(root) if f.endswith('.npy') and 'croppad' in f]

    for file in files:
        speaker_id = file.split('_')[0]
        speaker_files[speaker_id].append(file)

    return speaker_files


# 获取演员的后两位表示
def get_speaker_short_id(speaker_id):
    return speaker_id[2:]  # 取后两位，如1001->01, 1036->36


# 获取所有演员ID
speaker_ids = get_speaker_ids()
print(f"找到 {len(speaker_ids)} 个演员: {speaker_ids}")

# 按演员分组文件
speaker_files = group_files_by_speaker()

# 计算每个集合的演员数量
num_speakers = len(speaker_ids)
num_test_speakers = round(num_speakers * 0.15)  # 1.5/10 = 15%
num_val_speakers = round(num_speakers * 0.15)  # 1.5/10 = 15%
num_train_speakers = num_speakers - num_test_speakers - num_val_speakers

print(f"\n演员划分数量: 训练 {num_train_speakers}, 验证 {num_val_speakers}, 测试 {num_test_speakers}")

# 随机打乱演员ID
np.random.shuffle(speaker_ids)

# 划分演员到不同的集合
test_speakers = speaker_ids[:num_test_speakers]
val_speakers = speaker_ids[num_test_speakers:num_test_speakers + num_val_speakers]
train_speakers = speaker_ids[num_test_speakers + num_val_speakers:]

# 打印详细的演员分布信息
print(f"\n=== 演员分布详情 ===")
print(f"测试集演员 ({len(test_speakers)}人): {[get_speaker_short_id(s) for s in test_speakers]}")
print(f"验证集演员 ({len(val_speakers)}人): {[get_speaker_short_id(s) for s in val_speakers]}")
print(f"训练集演员 ({len(train_speakers)}人): {[get_speaker_short_id(s) for s in train_speakers]}")

# 打印完整的映射关系
print(f"\n=== 完整演员ID映射 ===")
print("演员ID\t后两位\t所属集合")
print("-" * 30)
for speaker_id in speaker_ids:
    short_id = get_speaker_short_id(speaker_id)
    if speaker_id in test_speakers:
        split_type = "测试集"
    elif speaker_id in val_speakers:
        split_type = "验证集"
    else:
        split_type = "训练集"
    print(f"{speaker_id}\t{short_id}\t{split_type}")

annotation_file = 'annotations.txt'

# 清空或创建标注文件
with open(annotation_file, 'w') as f:
    f.write('')

# 写入标注信息
for speaker_id, files in speaker_files.items():
    for video_file in files:
        label = video_file.split('_')[2]
        # 标签映射
        if label == "ANG":
            label = 1
        elif label == "DIS":
            label = 2
        elif label == "FEA":
            label = 3
        elif label == "HAP":
            label = 4
        elif label == "NEU":
            label = 5
        elif label == "SAD":
            label = 6

        label = str(label)
        audio_file = video_file.replace("facecroppad.npy", "croppad.wav")

        if speaker_id in train_speakers:
            split_type = 'training'
        elif speaker_id in val_speakers:
            split_type = 'validation'
        else:  # test_speakers
            split_type = 'testing'

        with open(annotation_file, 'a') as f:
            f.write(os.path.join(root, video_file) + ';' +
                    os.path.join(root, audio_file) + ';' +
                    label + ';' + split_type + '\n')

# 统计样本数量
train_samples = sum(len(speaker_files[s]) for s in train_speakers)
val_samples = sum(len(speaker_files[s]) for s in val_speakers)
test_samples = sum(len(speaker_files[s]) for s in test_speakers)
total_samples = train_samples + val_samples + test_samples

print(f"\n=== 样本统计 ===")
print(f"总样本数: {total_samples}")
print(f"训练集样本数: {train_samples} ({train_samples / total_samples * 100:.1f}%)")
print(f"验证集样本数: {val_samples} ({val_samples / total_samples * 100:.1f}%)")
print(f"测试集样本数: {test_samples} ({test_samples / total_samples * 100:.1f}%)")

print(f"\n标注文件已生成: {annotation_file}")

