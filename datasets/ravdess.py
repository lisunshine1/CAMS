# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa


def video_loader(video_dir_path):
    video = np.load(video_dir_path)    
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr)
    y = audios[0]
    return y, sr

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        filename, audiofilename, label, trainvaltest = line.split(';')        
        if trainvaltest.rstrip() != subset:
            continue
        
        sample = {'video_path': filename,                       
                  'audio_path': audiofilename, 
                  'label': int(label)-1}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,                   # 在opts中表明，对应的是生成的annotations.txt文件
                 subset,                            # 传入的training validation testing以获取训练集 验证集 测试集
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type = 'audiovisual', audio_transform=None):
        self.data = make_dataset(subset, annotation_path)   # 调用make_dataset函数创建数据集 注意哈，这个函数就已经将数据集划分为training validation testing
        # 将传入的参数保存为类属性
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type 

    # 用于获取指定索引处数据
    def __getitem__(self, index):
        target = self.data[index]['label']
                
        # 如果数据类型为video或者audiovisual，则加载对应视频数据
        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)

            # 如果要求空间变换，就对视频数据进行变换，然后将所有帧堆叠起来
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target

        # 如果数据类型为audio或者audiovisual，则加载对应音频数据
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=22050) 
            # 如果提供对应音频变换，就对音频数据进行变换，然后计算MFCC特征
            if self.audio_transform is not None:
                 self.audio_transform.randomize_parameters()
                 y = self.audio_transform(y)     
                 
            mfcc = get_mfccs(y, sr)            
            audio_features = mfcc 

            if self.data_type == 'audio':
                return audio_features, target

        # 如果数据类型为audiovisual，则返回音频特征、视频片段和目标标签
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  

    def __len__(self):
        return len(self.data)
