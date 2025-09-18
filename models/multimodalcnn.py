# # # # -*- coding: utf-8 -*-
# # # """
# # # Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
# # # """
# # #
# # # import torch
# # # import torch.nn as nn
# # # from models.modulator import Modulator
# # # from models.efficientface import LocalFeatureExtractor, InvertedResidual
# # # from models.transformer_timm import AttentionBlock, Attention
# # #
# # # def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
# # #     return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
# # #                                    nn.ReLU(inplace=True))
# # #
# # # class EfficientFaceTemporal(nn.Module): # EfficientFaceTemporal模型 这个模型是21年发表在AAI上的一个轻量级面部表情识别模型
# # #
# # #     def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
# # #         super(EfficientFaceTemporal, self).__init__()
# # #
# # #         if len(stages_repeats) != 3:
# # #             raise ValueError('expected stages_repeats as list of 3 positive ints')
# # #         if len(stages_out_channels) != 5:
# # #             raise ValueError('expected stages_out_channels as list of 5 positive ints')
# # #         self._stage_out_channels = stages_out_channels
# # #
# # #         # 这段代码定义了模型的一个卷积层，它包含一个2D卷积操作，一个批量化诡异操作，以及一个ReLu激活函数
# # #         input_channels = 3
# # #         output_channels = self._stage_out_channels[0]
# # #         self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
# # #                                    nn.BatchNorm2d(output_channels),
# # #                                    nn.ReLU(inplace=True),)
# # #
# # #         # 这段代码更新通道数量，便于下一层处理
# # #         input_channels = output_channels
# # #
# # #         # 定义最大池化
# # #         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# # #
# # #         # 定义模型三个结点，每个阶段包含一系列的InvertedResidual块
# # #         stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
# # #         for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
# # #             seq = [InvertedResidual(input_channels, output_channels, 2)]
# # #             for i in range(repeats - 1):
# # #                 seq.append(InvertedResidual(output_channels, output_channels, 1))
# # #             setattr(self, name, nn.Sequential(*seq))
# # #             input_channels = output_channels
# # #
# # #         # 局部特征提取器与调制器
# # #         self.local = LocalFeatureExtractor(29, 116, 1)
# # #         self.modulator = Modulator(116)
# # #
# # #         # 这几行代码定义了模型的第五个卷积层，包含一个2D卷积操作，一个批量归一化操作，以及一个ReLU激活函数
# # #         output_channels = self._stage_out_channels[-1]
# # #
# # #         self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
# # #                                    nn.BatchNorm2d(output_channels),
# # #                                    nn.ReLU(inplace=True),)
# # #
# # #         # 四行代码定义了四个1D卷积快
# # #         self.conv1d_0 = conv1d_block(output_channels, 64)
# # #         self.conv1d_1 = conv1d_block(64, 64)
# # #         self.conv1d_2 = conv1d_block(64, 128)
# # #         self.conv1d_3 = conv1d_block(128, 128)
# # #
# # #         # 图像辅助网络
# # #         # 四行代码定义了四个1D卷积快
# # #         self.conv1d_0_viusal_aux = conv1d_block(output_channels, 64)
# # #         self.conv1d_1_visual_aux = conv1d_block(64, 64)
# # #         self.conv1d_2_visual_aux = conv1d_block(64, 128)
# # #         self.conv1d_3_visual_aux = conv1d_block(128, 128)
# # #
# # #         # 线性分类器
# # #         self.classifier_1 = nn.Sequential(
# # #                 nn.Linear(128, num_classes),
# # #             )
# # #
# # #         # 将每个样本图像保存为类的一个属性
# # #         self.im_per_sample = im_per_sample
# # #
# # #     def forward_features(self, x):
# # #         x = self.conv1(x)
# # #         x = self.maxpool(x)
# # #         x = self.modulator(self.stage2(x)) + self.local(x)
# # #         x = self.stage3(x)
# # #         x = self.stage4(x)
# # #         x = self.conv5(x)
# # #         x = x.mean([2, 3]) #global average pooling
# # #         return x
# # #     # 对输入的视频x应用一些列的卷积，池化，调制器，以及局部特征提取器，然后对结果进行平均池化，并返回池化后结果 对应于图中EfficientFace提取图像特征
# # #
# # #     def forward_stage1(self, x):
# # #         #Getting samples per batch
# # #         assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
# # #         n_samples = x.shape[0] // self.im_per_sample
# # #         x = x.view(n_samples, self.im_per_sample, x.shape[1])
# # #         x = x.permute(0,2,1)
# # #         x = self.conv1d_0(x)
# # #         x = self.conv1d_1(x)
# # #         return x
# # #     # 检查x输入的形状是否满足预期，然后对x进行重塑与置换，再对x应用两个1D卷积快，并返回结果 对应图中图像分支前2个1D卷积
# # #
# # #     # 图像辅助网络
# # #     def forward_stage1_visual_aux(self, x):
# # #         #Getting samples per batch
# # #         assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
# # #         n_samples = x.shape[0] // self.im_per_sample
# # #         x = x.view(n_samples, self.im_per_sample, x.shape[1])
# # #         x = x.permute(0,2,1)
# # #         x = self.conv1d_0_viusal_aux(x)
# # #         x = self.conv1d_1_visual_aux(x)
# # #         return x
# # #
# # #     def forward_stage2(self, x):
# # #         x = self.conv1d_2(x)
# # #         x = self.conv1d_3(x)
# # #         return x
# # #     # 对输入x应用两个1D卷积块并返回结果 对应图中图像分支后2个1D卷积
# # #
# # #     # 图像辅助网络
# # #     def forward_stage2_visual_aux(self, x):
# # #         x = self.conv1d_2_visual_aux(x)
# # #         x = self.conv1d_3_visual_aux(x)
# # #         return x
# # #
# # #     def forward_classifier(self, x):
# # #         x = x.mean([-1]) #pooling accross temporal dimension
# # #         x1 = self.classifier_1(x)
# # #         return x1
# # #     # 对输入x进行平均池化，减少维度，然后将池化后x输入分类器，并返回分类器输出
# # #
# # #     def forward(self, x):
# # #         x = self.forward_features(x)
# # #         x = self.forward_stage1(x)
# # #         x = self.forward_stage2(x)
# # #         x = self.forward_classifier(x)
# # #         return x
# # #     # 对输入的x先进性forward_features进行特征提取，然后依次执行forward_stage1，forward_stage2和forward_classifier
# # #     # 这个就类比图中整个图像分支 （注意哈，相当于单模态的图像情感预测，没有注意机制，就是先调用forward_features提取图像特征，然后将特征通过
# # #     # forward_stage1和forward_stage2，对应图中4个1D卷积，最后通过forward_classifier线性层输出预测情感）
# # #
# # #
# # # # 接收两个参数，参数一模型，参数二路径 这里的路径是训练好模型的路径 实现的功能就是把训练好的模型参数加载进模型中
# # # def init_feature_extractor(model, path):
# # #     if path == 'None' or path is None:
# # #         return
# # #     checkpoint = torch.load(path, map_location=torch.device('cpu')) # 从path路径加载检查点
# # #     pre_trained_dict = checkpoint['state_dict'] # 从检查点中获取状态
# # #     pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
# # #     print('Initializing efficientnet') # 打印提示信息，表示初始化EfficientNet模型
# # #     model.load_state_dict(pre_trained_dict, strict=False) # 将预训练模型加载值model中
# # #
# # # # 参数一，类别数，参数二，人物，参数三，序列长度 创建一个EfficientFaceTemporal模型，并返回
# # # def get_model(num_classes, task, seq_length):
# # #     model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
# # #     return model
# # #
# # # # 参数一，输入通道数，参数二，输出通道数，参数三，卷积核大小，参数四，补偿，参数五，填充方法
# # # # 创建一个1D卷积块，包含一1D卷积层，一批量归一化层，一ReLU激活函数，一最大池化层
# # # def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
# # #     return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
# # #                                    nn.ReLU(inplace=True), nn.MaxPool1d(2,1))
# # #
# # # class AudioCNNPool(nn.Module): # AudioCNNPool模型实例，用于处理音频数据
# # #
# # #     def __init__(self, num_classes=8):
# # #         super(AudioCNNPool, self).__init__()
# # #
# # #         # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
# # #         input_channels = 10
# # #         self.conv1d_0 = conv1d_block_audio(input_channels, 64)
# # #         self.conv1d_1 = conv1d_block_audio(64, 128)
# # #         self.conv1d_2 = conv1d_block_audio(128, 256)
# # #         self.conv1d_3 = conv1d_block_audio(256, 128)
# # #
# # #         # 音频辅助网络
# # #         # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
# # #         input_channels = 10
# # #         self.conv1d_0_audio_aux = conv1d_block_audio(input_channels, 64)
# # #         self.conv1d_1_audio_aux = conv1d_block_audio(64, 128)
# # #         self.conv1d_2_audio_aux = conv1d_block_audio(128, 256)
# # #         self.conv1d_3_audio_aux = conv1d_block_audio(256, 128)
# # #
# # #         # 定义一个线性分类层
# # #         self.classifier_1 = nn.Sequential(
# # #                 nn.Linear(128, num_classes),
# # #             )
# # #
# # #     # 四个卷积，然后分类器，输出音频情感预测
# # #     def forward(self, x):
# # #         x = self.forward_stage1(x)
# # #         x = self.forward_stage2(x)
# # #         x = self.forward_classifier(x)
# # #         return x
# # #
# # #
# # #     def forward_stage1(self,x):
# # #         x = self.conv1d_0(x)
# # #         x = self.conv1d_1(x)
# # #         return x
# # #
# # #     # 音频辅助网络
# # #     def forward_stage1_audio_aux(self,x):
# # #         x = self.conv1d_0_audio_aux(x)
# # #         x = self.conv1d_1_audio_aux(x)
# # #         return x
# # #
# # #     def forward_stage2(self,x):
# # #         x = self.conv1d_2(x)
# # #         x = self.conv1d_3(x)
# # #         return x
# # #
# # #     # 音频辅助网络
# # #     def forward_stage2_audio_aux(self,x):
# # #         x = self.conv1d_2_audio_aux(x)
# # #         x = self.conv1d_3_audio_aux(x)
# # #         return x
# # #
# # #     def forward_classifier(self, x):
# # #         x = x.mean([-1]) #pooling accross temporal dimension
# # #         x1 = self.classifier_1(x)
# # #         return x1
# # #
# # #
# # #
# # #
# # # class MultiModalCNN(nn.Module):
# # #     def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1): # 这里的pretr_ef就是预训练EfficientFace_Trained_on_AffectNet7.pth.tar
# # #         super(MultiModalCNN, self).__init__()
# # #         assert fusion in ['ia', 'it', 'lt'], print('Unsupported fusion method: {}'.format(fusion)) # 这行代码检查fusion参数是否在['ia', 'it', 'lt']中。如果不在，就会打印一条错误消息。
# # #
# # #         self.audio_model = AudioCNNPool(num_classes=num_classes) # 这行代码创建了一个AudioCNNPool模型实例，用于处理音频数据。
# # #         self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length) # 这行代码创建了一个EfficientFaceTemporal模型实例，用于处理视觉数据
# # #
# # #         init_feature_extractor(self.visual_model, pretr_ef) # 这行代码使用预训练的模型初始化EfficientFaceTemporal模型
# # #
# # #         e_dim = 128 # 嵌入维度
# # #         input_dim_video = 128 # 视频输入维度
# # #         input_dim_audio = 128 # 音频的输入维度
# # #         self.fusion=fusion # 融合方法
# # #
# # #         # 通过不同的融合方法，创建对应注意力块
# # #         # Transformer对应为AttentionBlock，而Attention机制对应为Attention()
# # #         # 这里的lt就表示后期融合Transformer it表示中期融合Transformer
# # #         # if fusion in ['lt', 'it']:
# # #         #     if fusion  == 'lt':
# # #         #         self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
# # #         #         self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
# # #         #     elif fusion == 'it':
# # #         #         input_dim_video = input_dim_video // 2
# # #         #         self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
# # #         #         self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
# # #         # # 这里ia表示中期融合的注意机制
# # #         # elif fusion in ['ia']:
# # #         #     input_dim_video = input_dim_video // 2
# # #         #
# # #         #     self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
# # #         #     self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
# # #
# # #
# # #         # 后期Transformer
# # #         self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
# # #                                  num_heads=num_heads)
# # #         self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
# # #                                  num_heads=num_heads)
# # #
# # #         # 中期自注意机制
# # #         self.aa = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
# # #                                  num_heads=num_heads)
# # #         self.vv = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
# # #                                  num_heads=num_heads)
# # #
# # #         # 中期Attention
# # #         input_dim_video = input_dim_video // 2
# # #         self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
# # #                              num_heads=num_heads)
# # #         self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
# # #                              num_heads=num_heads)
# # #
# # #
# # #         # self.classifier_1 = nn.Sequential( # 创建一个线性分类器
# # #         #             nn.Linear(e_dim*2, num_classes), # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
# # #         #         )
# # #
# # #         self.classifier_1 = nn.Sequential( # 创建一个线性分类器
# # #                     nn.Linear(e_dim*4, num_classes), # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
# # #                 )
# # #
# # #
# # #
# # #     def forward(self, x_audio, x_visual): # 是MultiModalCNN类的主要方法，它通过融合方法fusion来决定使用哪一个前线传播
# # #
# # #         return self.forward_own(x_audio, x_visual)
# # #         # if self.fusion == 'lt':
# # #         #     return self.forward_transformer(x_audio, x_visual)
# # #         #
# # #         # elif self.fusion == 'ia':
# # #         #     return self.forward_feature_2(x_audio, x_visual)
# # #         #
# # #         # elif self.fusion == 'it':
# # #         #     return self.forward_feature_3(x_audio, x_visual)
# # #
# # #     # PS：输入的音频特征与图像特征都是128维度
# # #
# # #     # 这个是中期Transformer的前向传播
# # #     def forward_feature_3(self, x_audio, x_visual):
# # #         x_audio = self.audio_model.forward_stage1(x_audio) # 将音频数据输入2个1D卷积中，得到128维度输出
# # #         x_visual = self.visual_model.forward_features(x_visual) # 将图像数据输入特征提取模块
# # #         x_visual = self.visual_model.forward_stage1(x_visual) # 在经过2个1D卷积，得到64维输出
# # #
# # #         # 假设有一个3维张量，形状是(2, 3, 4)，其中的2表示由2个矩阵，每个矩阵有3行4列，如果使用permute(0, 2, 1)后，那么张量变为(2, 4, 3)
# # #         proj_x_a = x_audio.permute(0,2,1)
# # #         proj_x_v = x_visual.permute(0,2,1)
# # #
# # #         # 这两行使用Transformer机制
# # #         h_av = self.av1(proj_x_v, proj_x_a) # 以音频为k v，视频为q，属于视频分支
# # #         h_va = self.va1(proj_x_a, proj_x_v) # 以视频为k v，音频为q，属于音频分支
# # #
# # #         # 对于Transformer的输出，恢复其形状
# # #         h_av = h_av.permute(0,2,1)
# # #         h_va = h_va.permute(0,2,1)
# # #
# # #         # 将Transformer的输出添加到分支上
# # #         x_audio = h_av+x_audio
# # #         x_visual = h_va + x_visual
# # #
# # #         # 将音频与图像特征，通过后面2层1D卷积
# # #         x_audio = self.audio_model.forward_stage2(x_audio)
# # #         x_visual = self.visual_model.forward_stage2(x_visual)
# # #
# # #         # 进行池化操作
# # #         audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
# # #         video_pooled = x_visual.mean([-1])
# # #
# # #         # 将池化后的音频特征与图像特征拼接
# # #         x = torch.cat((audio_pooled, video_pooled), dim=-1)
# # #
# # #         # 将拼接特征送入分类器中，进行情绪预测
# # #         x1 = self.classifier_1(x)
# # #         return x1
# # #
# # #     # 这个是中期注意机制的前向传播
# # #     # def forward_feature_2(self, x_audio, x_visual):
# # #     #     x_audio = self.audio_model.forward_stage1(x_audio) # 将音频特征输入2层1D卷积中，得到128维输出
# # #     #     x_visual = self.visual_model.forward_features(x_visual) # 将图像通过EfficientFace进行特征提取
# # #     #     x_visual = self.visual_model.forward_stage1(x_visual) # 将提取图像特征送入2层1D卷积中得到64维输出
# # #     #     print("stage1 audio = ", x_audio.shape)             # ([40, 128, 150])
# # #     #     print("stage1 visual =", x_visual.shape)            # ([40, 64, 15])
# # #     #     proj_x_a = x_audio.permute(0,2,1) # 调整矩阵行与列
# # #     #     proj_x_v = x_visual.permute(0,2,1) # 调整矩阵行与列
# # #     #     print("changeshape audio = ", proj_x_a.shape)       # ([40, 150, 128])
# # #     #     print("changeshape visual =", proj_x_v.shape)       # ([40, 15, 64])
# # #     #     _, h_av = self.av1(proj_x_v, proj_x_a) # 使用注意机制，属于音频分支
# # #     #     _, h_va = self.va1(proj_x_a, proj_x_v) # 使用注意机制，属于视频分支
# # #     #
# # #     #     if h_av.size(1) > 1: #if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
# # #     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# # #     #
# # #     #     h_av = h_av.sum([-2]) # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
# # #     #                           #                                                     [3 4 4 ]
# # #     #                           # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
# # #     #
# # #     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
# # #     #     if h_va.size(1) > 1: #if more than 1 head, take average
# # #     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# # #     #
# # #     #     h_va = h_va.sum([-2])
# # #     #
# # #     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
# # #     #     x_audio = h_va*x_audio
# # #     #     x_visual = h_av*x_visual
# # #     #     print("afterattention audio = ", x_audio.shape)     # ([40, 128, 150])
# # #     #     print("afterattention visual =", x_visual.shape)    # ([40, 64, 15])
# # #     #     # 然后经过后面的2层1D卷积
# # #     #     x_audio = self.audio_model.forward_stage2(x_audio)
# # #     #     x_visual = self.visual_model.forward_stage2(x_visual)
# # #     #     print("stage2 audio = ", x_audio.shape)             # ([40, 128, 144])
# # #     #     print("stage2 visual =", x_visual.shape)            # ([40, 128, 15])
# # #     #     # 池化层
# # #     #     audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
# # #     #     video_pooled = x_visual.mean([-1])
# # #     #     print("pooled audio = ", audio_pooled.shape)        # ([40, 128])
# # #     #     print("pooled visual =", video_pooled.shape)        # ([40, 128])
# # #     #
# # #     #     # 拼接特征
# # #     #     x = torch.cat((audio_pooled, video_pooled), dim=-1)
# # #     #
# # #     #     # 进行分类
# # #     #     x1 = self.classifier_1(x)
# # #     #     return x1
# # #
# # #     def forward_feature_2(self, x_audio, x_visual):
# # #         x_audio = self.audio_model.forward_stage1(x_audio)
# # #         x_visual = self.visual_model.forward_features(x_visual)
# # #         x_visual = self.visual_model.forward_stage1(x_visual)
# # #
# # #         proj_x_a = x_audio.permute(0, 2, 1)
# # #         proj_x_v = x_visual.permute(0, 2, 1)
# # #
# # #         _, h_av = self.av1(proj_x_v, proj_x_a)
# # #         _, h_va = self.va1(proj_x_a, proj_x_v)
# # #
# # #         if h_av.size(1) > 1:  # if more than 1 head, take average
# # #             h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# # #
# # #         h_av = h_av.sum([-2])
# # #
# # #         if h_va.size(1) > 1:  # if more than 1 head, take average
# # #             h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# # #
# # #         h_va = h_va.sum([-2])
# # #
# # #         x_audio = h_va * x_audio
# # #         x_visual = h_av * x_visual
# # #
# # #         x_audio = self.audio_model.forward_stage2(x_audio)
# # #         x_visual = self.visual_model.forward_stage2(x_visual)
# # #
# # #         audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
# # #         video_pooled = x_visual.mean([-1])
# # #
# # #         x = torch.cat((audio_pooled, video_pooled), dim=-1)
# # #
# # #         x1 = self.classifier_1(x)
# # #         return x1
# # #
# # #     # 这个是后期Transformer的前向传播
# # #     def forward_transformer(self, x_audio, x_visual):
# # #         print("进入后期Transformer\n")
# # #         x_audio = self.audio_model.forward_stage1(x_audio)
# # #         print("x_audio.shape stage1 = ", x_audio.shape)                 # ([40, 128, 150]) 音频1
# # #         proj_x_a = self.audio_model.forward_stage2(x_audio) # 得到128维输出
# # #         print("x_audio.shape stage2 = ", proj_x_a.shape)                # ([40, 128, 144]) 音频2
# # #
# # #         x_visual = self.visual_model.forward_features(x_visual)
# # #         x_visual = self.visual_model.forward_stage1(x_visual)
# # #         print("x_visual.shape stage1 = ", x_visual.shape)               # ([40, 64, 15]) 视频1
# # #         proj_x_v = self.visual_model.forward_stage2(x_visual) # 得到128维输出
# # #         print("x_visual.shape stage2 = ", proj_x_v.shape)               # ([40, 128, 15]) 视频2
# # #
# # #         proj_x_a = proj_x_a.permute(0, 2, 1)
# # #         proj_x_v = proj_x_v.permute(0, 2, 1)
# # #         print("转换形状之后\n")
# # #         print("x_audio.shape shape = ", proj_x_a.shape)                 # ([40, 144, 128]) 音频
# # #         print("x_visual.shape shape = ", proj_x_v.shape)                # ([40, 15, 128]) 视频
# # #         h_av = self.av(proj_x_v, proj_x_a) # Transformer
# # #         h_va = self.va(proj_x_a, proj_x_v)
# # #         print("经过Transformer后\n")
# # #         print("h_av.shape = ", h_av.shape)                              # ([40, 144, 128]) 音频
# # #         print("h_va.shape = ", h_av.shape)                              # ([40, 144, 128]) 视频
# # #         # 池化层
# # #         audio_pooled = h_av.mean([1]) #mean accross temporal dimension
# # #         video_pooled = h_va.mean([1])
# # #         print("池化后\n")
# # #         print("audio_pooled.shape = ", audio_pooled.shape)              # ([40, 128]) 音频
# # #         print("video_pooled.shape = ", video_pooled.shape)              # ([40, 128]) 视频
# # #         # 拼接特征
# # #         x = torch.cat((audio_pooled, video_pooled), dim=-1)
# # #         print("拼接后\n")
# # #         print("x.shape = ", x.shape)                                    # ([40, 256])
# # #
# # #         # 情感预测
# # #         x1 = self.classifier_1(x)
# # #         print("预测后\n")
# # #         print("x1.shape= ", x1.shape)                                   # ([40, 8])
# # #         return x1
# # #
# # #     # 自己改的，想要Attention中期 单头注意机制 + Transformer后期 + 两个单模态，最后改下线性层的输入维度为e_dim * 4就好 然后把4个特征拼接起来
# # #     # 但是希望在两个单模态的分支上，能够添加损失函数与中间的分支做交互，因为目的是为了补偿
# # #     # def forward_own(self, x_audio, x_visual):
# # #     #     x_audio_aux = x_audio # 音频辅助网络
# # #     #
# # #     #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
# # #     #     # print("音频 stage1 = ", x_audio.shape)
# # #     #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
# # #     #
# # #     #     x_visual_aux = x_visual # 图像辅助网络
# # #     #
# # #     #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
# # #     #     # print("视频 stage1 = ", x_visual.shape)
# # #     #
# # #     #     # 辅助网络
# # #     #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
# # #     #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
# # #     #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
# # #     #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
# # #     #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
# # #     #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
# # #     #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
# # #     #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
# # #     #
# # #     #     # 辅助网池化层
# # #     #     audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
# # #     #     visual_aux_pooled = x_visual_aux.mean([-1])
# # #     #     # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)
# # #     #     # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)
# # #     #
# # #     #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
# # #     #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
# # #     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
# # #     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
# # #     #
# # #     #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
# # #     #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
# # #     #
# # #     #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
# # #     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# # #     #
# # #     #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
# # #     #     #                                                     [3 4 4 ]
# # #     #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
# # #     #
# # #     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
# # #     #     if h_va.size(1) > 1:  # if more than 1 head, take average
# # #     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# # #     #
# # #     #     h_va = h_va.sum([-2])
# # #     #
# # #     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
# # #     #     x_audio = h_va * x_audio
# # #     #     x_visual = h_av * x_visual
# # #     #     # print("音频 注意机制 x_audio = ", x_audio.shape)
# # #     #     # print("图像 注意机制 x_visual = ", x_visual.shape)
# # #     #
# # #     #     # 然后经过后面的2层1D卷积
# # #     #     x_audio = self.audio_model.forward_stage2(x_audio)
# # #     #     x_visual = self.visual_model.forward_stage2(x_visual)
# # #     #     # print("音频 stage2 x_audio = ", x_audio.shape)
# # #     #     # print("图像 stage2 x_visual = ", x_visual.shape)
# # #     #
# # #     #     # 使用后期Transformer
# # #     #     proj_x_a = x_audio.permute(0, 2, 1)
# # #     #     proj_x_v = x_visual.permute(0, 2, 1)
# # #     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
# # #     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
# # #     #     h_av = self.av(proj_x_v, proj_x_a) # Transformer
# # #     #     h_va = self.va(proj_x_a, proj_x_v)
# # #     #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
# # #     #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
# # #     #
# # #     #     # 池化层
# # #     #     audio_pooled = h_av.mean([1])  # mean accross temporal dimension
# # #     #     video_pooled = h_va.mean([1])
# # #     #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
# # #     #     # print("视频 池化 video_pooled = ", video_pooled.shape)
# # #     #
# # #     #     # 拼接特征
# # #     #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
# # #     #     x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
# # #     #
# # #     #     # 进行分类
# # #     #     x1 = self.classifier_1(x)
# # #     #     return x1
# # #
# # #     # 辅助网络有包含中期Transformer块
# # #     def forward_own(self, x_audio, x_visual):
# # #         x_audio_aux = x_audio # 音频辅助网络
# # #
# # #         x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
# # #         # print("音频 stage1 = ", x_audio.shape)
# # #         x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
# # #
# # #         x_visual_aux = x_visual # 图像辅助网络
# # #
# # #         x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
# # #         # print("视频 stage1 = ", x_visual.shape)
# # #
# # #         # 辅助网络
# # #         x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
# # #         # print("辅助音频 stage1 = ", x_audio_aux.shape)
# # #         x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
# # #         # print("辅助音频 stage2 = ", x_audio_aux.shape)
# # #         x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
# # #         # print("辅助视频 stage1 = ", x_visual_aux.shape)
# # #         x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
# # #         # print("辅助视频 stage2 = ", x_visual_aux.shape)
# # #
# # #         # 辅助网络转换
# # #         x_audio_aux = x_audio_aux.permute(0, 2, 1)
# # #         x_visual_aux = x_visual_aux.permute(0, 2, 1)
# # #         # print("转换形状之后\n")
# # #         # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
# # #         # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
# # #
# # #         # 辅助网络后期自注意
# # #         x_audio_aux = self.av(x_audio_aux, x_audio_aux) # Transformer
# # #         x_visual_aux = self.va(x_visual_aux, x_visual_aux)
# # #         # print("经过Transformer后\n")
# # #         # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
# # #         # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
# # #
# # #         # 辅助网络转换
# # #         x_audio_aux = x_audio_aux.permute(0, 2, 1)
# # #         x_visual_aux = x_visual_aux.permute(0, 2, 1)
# # #         # print("转换形状之后\n")
# # #         # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
# # #         # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
# # #
# # #         # 辅助网池化层
# # #         audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
# # #         visual_aux_pooled = x_visual_aux.mean([-1])
# # #         # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
# # #         # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
# # #
# # #         proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
# # #         proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
# # #         # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
# # #         # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
# # #
# # #         _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
# # #         _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
# # #
# # #         if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
# # #             h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# # #
# # #         h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
# # #         #                                                     [3 4 4 ]
# # #         # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
# # #
# # #         # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
# # #         if h_va.size(1) > 1:  # if more than 1 head, take average
# # #             h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# # #
# # #         h_va = h_va.sum([-2])
# # #
# # #         # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
# # #         x_audio = h_va * x_audio
# # #         x_visual = h_av * x_visual
# # #         # print("音频 注意机制 x_audio = ", x_audio.shape)
# # #         # print("图像 注意机制 x_visual = ", x_visual.shape)
# # #
# # #         # 然后经过后面的2层1D卷积
# # #         x_audio = self.audio_model.forward_stage2(x_audio)
# # #         x_visual = self.visual_model.forward_stage2(x_visual)
# # #         # print("音频 stage2 x_audio = ", x_audio.shape)
# # #         # print("图像 stage2 x_visual = ", x_visual.shape)
# # #
# # #         # 使用后期Transformer
# # #         proj_x_a = x_audio.permute(0, 2, 1)
# # #         proj_x_v = x_visual.permute(0, 2, 1)
# # #         # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
# # #         # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
# # #         h_av = self.av(proj_x_v, proj_x_a) # Transformer
# # #         h_va = self.va(proj_x_a, proj_x_v)
# # #         # print("音频 Transformer之后 x_audio = ", h_av.shape)
# # #         # print("图像 Transformer之后 x_visual = ", h_va.shape)
# # #
# # #         # 池化层
# # #         audio_pooled = h_av.mean([1])  # mean accross temporal dimension
# # #         video_pooled = h_va.mean([1])
# # #         # print("音频 池化 audio_pooled = ", audio_pooled.shape)
# # #         # print("视频 池化 video_pooled = ", video_pooled.shape)
# # #
# # #         # 拼接特征
# # #         # x = torch.cat((audio_pooled, video_pooled), dim=-1)
# # #         x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
# # #
# # #         # 进行分类
# # #         x1 = self.classifier_1(x)
# # #         return x1
# # #
# # #
# # #
# # # -*- coding: utf-8 -*-
# # """
# # Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
# # """
# #
# # import torch
# # import torch.nn as nn
# # from models.modulator import Modulator
# # from models.efficientface import LocalFeatureExtractor, InvertedResidual
# # from models.transformer_timm import AttentionBlock, Attention
# # from models.co_attention import CoAttention
# #
# #
# # def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
# #     return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
# #                          nn.BatchNorm1d(out_channels),
# #                          nn.ReLU(inplace=True))
# #
# #
# # class EfficientFaceTemporal(nn.Module):  # EfficientFaceTemporal模型 这个模型是21年发表在AAI上的一个轻量级面部表情识别模型
# #
# #     def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
# #         super(EfficientFaceTemporal, self).__init__()
# #
# #         if len(stages_repeats) != 3:
# #             raise ValueError('expected stages_repeats as list of 3 positive ints')
# #         if len(stages_out_channels) != 5:
# #             raise ValueError('expected stages_out_channels as list of 5 positive ints')
# #         self._stage_out_channels = stages_out_channels
# #
# #         # 这段代码定义了模型的一个卷积层，它包含一个2D卷积操作，一个批量化诡异操作，以及一个ReLu激活函数
# #         input_channels = 3
# #         output_channels = self._stage_out_channels[0]
# #         self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
# #                                    nn.BatchNorm2d(output_channels),
# #                                    nn.ReLU(inplace=True), )
# #
# #         # 这段代码更新通道数量，便于下一层处理
# #         input_channels = output_channels
# #
# #         # 定义最大池化
# #         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# #
# #         # 定义模型三个结点，每个阶段包含一系列的InvertedResidual块
# #         stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
# #         for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
# #             seq = [InvertedResidual(input_channels, output_channels, 2)]
# #             for i in range(repeats - 1):
# #                 seq.append(InvertedResidual(output_channels, output_channels, 1))
# #             setattr(self, name, nn.Sequential(*seq))
# #             input_channels = output_channels
# #
# #         # 局部特征提取器与调制器
# #         self.local = LocalFeatureExtractor(29, 116, 1)
# #         self.modulator = Modulator(116)
# #
# #         # 这几行代码定义了模型的第五个卷积层，包含一个2D卷积操作，一个批量归一化操作，以及一个ReLU激活函数
# #         output_channels = self._stage_out_channels[-1]
# #
# #         self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
# #                                    nn.BatchNorm2d(output_channels),
# #                                    nn.ReLU(inplace=True), )
# #
# #         # 四行代码定义了四个1D卷积快
# #         self.conv1d_0 = conv1d_block(output_channels, 64)
# #         self.conv1d_1 = conv1d_block(64, 64)
# #         self.conv1d_2 = conv1d_block(64, 128)
# #         self.conv1d_3 = conv1d_block(128, 128)
# #
# #         # 图像辅助网络
# #         # 四行代码定义了四个1D卷积快
# #         self.conv1d_0_viusal_aux = conv1d_block(output_channels, 64)
# #         self.conv1d_1_visual_aux = conv1d_block(64, 64)
# #         self.conv1d_2_visual_aux = conv1d_block(64, 128)
# #         self.conv1d_3_visual_aux = conv1d_block(128, 128)
# #
# #         # 线性分类器
# #         self.classifier_1 = nn.Sequential(
# #             nn.Linear(128, num_classes),
# #         )
# #
# #         # 将每个样本图像保存为类的一个属性
# #         self.im_per_sample = im_per_sample
# #
# #     def forward_features(self, x):
# #         x = self.conv1(x)
# #         x = self.maxpool(x)
# #         x = self.modulator(self.stage2(x)) + self.local(x)
# #         x = self.stage3(x)
# #         x = self.stage4(x)
# #         x = self.conv5(x)
# #         x = x.mean([2, 3])  # global average pooling
# #         return x
# #
# #     # 对输入的视频x应用一些列的卷积，池化，调制器，以及局部特征提取器，然后对结果进行平均池化，并返回池化后结果 对应于图中EfficientFace提取图像特征
# #
# #     def forward_stage1(self, x):
# #         # Getting samples per batch
# #         assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
# #         n_samples = x.shape[0] // self.im_per_sample
# #         x = x.view(n_samples, self.im_per_sample, x.shape[1])
# #         x = x.permute(0, 2, 1)
# #         x = self.conv1d_0(x)
# #         x = self.conv1d_1(x)
# #         return x
# #
# #     # 检查x输入的形状是否满足预期，然后对x进行重塑与置换，再对x应用两个1D卷积快，并返回结果 对应图中图像分支前2个1D卷积
# #
# #     # 图像辅助网络
# #     def forward_stage1_visual_aux(self, x):
# #         # Getting samples per batch
# #         assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
# #         n_samples = x.shape[0] // self.im_per_sample
# #         x = x.view(n_samples, self.im_per_sample, x.shape[1])
# #         x = x.permute(0, 2, 1)
# #         x = self.conv1d_0_viusal_aux(x)
# #         x = self.conv1d_1_visual_aux(x)
# #         return x
# #
# #     def forward_stage2(self, x):
# #         x = self.conv1d_2(x)
# #         x = self.conv1d_3(x)
# #         return x
# #
# #     # 对输入x应用两个1D卷积块并返回结果 对应图中图像分支后2个1D卷积
# #
# #     # 图像辅助网络
# #     def forward_stage2_visual_aux(self, x):
# #         x = self.conv1d_2_visual_aux(x)
# #         x = self.conv1d_3_visual_aux(x)
# #         return x
# #
# #     def forward_classifier(self, x):
# #         x = x.mean([-1])  # pooling accross temporal dimension
# #         x1 = self.classifier_1(x)
# #         return x1
# #
# #     # 对输入x进行平均池化，减少维度，然后将池化后x输入分类器，并返回分类器输出
# #
# #     def forward(self, x):
# #         x = self.forward_features(x)
# #         x = self.forward_stage1(x)
# #         x = self.forward_stage2(x)
# #         x = self.forward_classifier(x)
# #         return x
# #     # 对输入的x先进性forward_features进行特征提取，然后依次执行forward_stage1，forward_stage2和forward_classifier
# #     # 这个就类比图中整个图像分支 （注意哈，相当于单模态的图像情感预测，没有注意机制，就是先调用forward_features提取图像特征，然后将特征通过
# #     # forward_stage1和forward_stage2，对应图中4个1D卷积，最后通过forward_classifier线性层输出预测情感）
# #
# #
# # # 接收两个参数，参数一模型，参数二路径 这里的路径是训练好模型的路径 实现的功能就是把训练好的模型参数加载进模型中
# # def init_feature_extractor(model, path):
# #     if path == 'None' or path is None:
# #         return
# #     checkpoint = torch.load(path, map_location=torch.device('cpu'))  # 从path路径加载检查点
# #     pre_trained_dict = checkpoint['state_dict']  # 从检查点中获取状态
# #     pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
# #     print('Initializing efficientnet')  # 打印提示信息，表示初始化EfficientNet模型
# #     model.load_state_dict(pre_trained_dict, strict=False)  # 将预训练模型加载值model中
# #
# #
# # # 参数一，类别数，参数二，人物，参数三，序列长度 创建一个EfficientFaceTemporal模型，并返回
# # def get_model(num_classes, task, seq_length):
# #     model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
# #     return model
# #
# #
# # # 参数一，输入通道数，参数二，输出通道数，参数三，卷积核大小，参数四，补偿，参数五，填充方法
# # # 创建一个1D卷积块，包含一1D卷积层，一批量归一化层，一ReLU激活函数，一最大池化层
# # def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
# #     return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
# #                          nn.BatchNorm1d(out_channels),
# #                          nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))
# #
# #
# # class AudioCNNPool(nn.Module):  # AudioCNNPool模型实例，用于处理音频数据
# #
# #     def __init__(self, num_classes=8):
# #         super(AudioCNNPool, self).__init__()
# #
# #         # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
# #         input_channels = 10
# #         self.conv1d_0 = conv1d_block_audio(input_channels, 64)
# #         self.conv1d_1 = conv1d_block_audio(64, 128)
# #         self.conv1d_2 = conv1d_block_audio(128, 256)
# #         self.conv1d_3 = conv1d_block_audio(256, 128)
# #
# #         # 音频辅助网络
# #         # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
# #         input_channels = 10
# #         self.conv1d_0_audio_aux = conv1d_block_audio(input_channels, 64)
# #         self.conv1d_1_audio_aux = conv1d_block_audio(64, 128)
# #         self.conv1d_2_audio_aux = conv1d_block_audio(128, 256)
# #         self.conv1d_3_audio_aux = conv1d_block_audio(256, 128)
# #
# #         # 定义一个线性分类层
# #         self.classifier_1 = nn.Sequential(
# #             nn.Linear(128, num_classes),
# #         )
# #
# #     # 四个卷积，然后分类器，输出音频情感预测
# #     def forward(self, x):
# #         x = self.forward_stage1(x)
# #         x = self.forward_stage2(x)
# #         x = self.forward_classifier(x)
# #         return x
# #
# #     def forward_stage1(self, x):
# #         x = self.conv1d_0(x)
# #         x = self.conv1d_1(x)
# #         return x
# #
# #     # 音频辅助网络
# #     def forward_stage1_audio_aux(self, x):
# #         x = self.conv1d_0_audio_aux(x)
# #         x = self.conv1d_1_audio_aux(x)
# #         return x
# #
# #     def forward_stage2(self, x):
# #         x = self.conv1d_2(x)
# #         x = self.conv1d_3(x)
# #         return x
# #
# #     # 音频辅助网络
# #     def forward_stage2_audio_aux(self, x):
# #         x = self.conv1d_2_audio_aux(x)
# #         x = self.conv1d_3_audio_aux(x)
# #         return x
# #
# #     def forward_classifier(self, x):
# #         x = x.mean([-1])  # pooling accross temporal dimension
# #         x1 = self.classifier_1(x)
# #         return x1
# #
# #
# # class MultiModalCNN(nn.Module):
# #     def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None',
# #                  num_heads=1):  # 这里的pretr_ef就是预训练EfficientFace_Trained_on_AffectNet7.pth.tar
# #         super(MultiModalCNN, self).__init__()
# #         assert fusion in ['ia', 'it', 'lt'], print(
# #             'Unsupported fusion method: {}'.format(fusion))  # 这行代码检查fusion参数是否在['ia', 'it', 'lt']中。如果不在，就会打印一条错误消息。
# #
# #         self.audio_model = AudioCNNPool(num_classes=num_classes)  # 这行代码创建了一个AudioCNNPool模型实例，用于处理音频数据。
# #         self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes,
# #                                                   seq_length)  # 这行代码创建了一个EfficientFaceTemporal模型实例，用于处理视觉数据
# #
# #         init_feature_extractor(self.visual_model, pretr_ef)  # 这行代码使用预训练的模型初始化EfficientFaceTemporal模型
# #
# #         e_dim = 128  # 嵌入维度
# #         input_dim_video = 128  # 视频输入维度
# #         input_dim_audio = 128  # 音频的输入维度
# #         self.fusion = fusion  # 融合方法
# #
# #         # 通过不同的融合方法，创建对应注意力块
# #         # Transformer对应为AttentionBlock，而Attention机制对应为Attention()
# #         # 这里的lt就表示后期融合Transformer it表示中期融合Transformer
# #         # if fusion in ['lt', 'it']:
# #         #     if fusion  == 'lt':
# #         #         self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
# #         #         self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
# #         #     elif fusion == 'it':
# #         #         input_dim_video = input_dim_video // 2
# #         #         self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
# #         #         self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
# #         # # 这里ia表示中期融合的注意机制
# #         # elif fusion in ['ia']:
# #         #     input_dim_video = input_dim_video // 2
# #         #
# #         #     self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
# #         #     self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
# #
# #         # 后期Transformer
# #         self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
# #                                  num_heads=num_heads)
# #         self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
# #                                  num_heads=num_heads)
# #
# #         # 中期自注意机制
# #         self.aa1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
# #                                   num_heads=num_heads)
# #         self.vv1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
# #                                   num_heads=num_heads)
# #
# #         # 中期Attention
# #         input_dim_video = input_dim_video // 2
# #         self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
# #                              num_heads=num_heads)
# #         self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
# #                              num_heads=num_heads)
# #
# #         # # 辅助网络Attention
# #         self.aa = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
# #                             num_heads=num_heads)
# #         self.vv = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_video, out_dim=input_dim_video,
# #                             num_heads=num_heads)
# #
# #         # 主干网络的co-attention机制
# #         self.co_av = CoAttention(input_dim=128, hidden_dim=64)
# #
# #         # self.classifier_1 = nn.Sequential( # 创建一个线性分类器
# #         #             nn.Linear(e_dim*2, num_classes), # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
# #         #         )
# #
# #         self.classifier_1 = nn.Sequential(  # 创建一个线性分类器
# #             nn.Linear(e_dim * 3, num_classes),  # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
# #         )
# #
# #     def forward(self, x_audio, x_visual):  # 是MultiModalCNN类的主要方法，它通过融合方法fusion来决定使用哪一个前线传播
# #
# #         return self.forward_own(x_audio, x_visual)
# #         # if self.fusion == 'lt':
# #         #     return self.forward_transformer(x_audio, x_visual)
# #         #
# #         # elif self.fusion == 'ia':
# #         #     return self.forward_feature_2(x_audio, x_visual)
# #         #
# #         # elif self.fusion == 'it':
# #         #     return self.forward_feature_3(x_audio, x_visual)
# #
# #     # PS：输入的音频特征与图像特征都是128维度
# #
# #     # 这个是中期Transformer的前向传播
# #     def forward_feature_3(self, x_audio, x_visual):
# #         x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频数据输入2个1D卷积中，得到128维度输出
# #         x_visual = self.visual_model.forward_features(x_visual)  # 将图像数据输入特征提取模块
# #         x_visual = self.visual_model.forward_stage1(x_visual)  # 在经过2个1D卷积，得到64维输出
# #
# #         # 假设有一个3维张量，形状是(2, 3, 4)，其中的2表示由2个矩阵，每个矩阵有3行4列，如果使用permute(0, 2, 1)后，那么张量变为(2, 4, 3)
# #         proj_x_a = x_audio.permute(0, 2, 1)
# #         proj_x_v = x_visual.permute(0, 2, 1)
# #
# #         # 这两行使用Transformer机制
# #         h_av = self.av1(proj_x_v, proj_x_a)  # 以音频为k v，视频为q，属于视频分支
# #         h_va = self.va1(proj_x_a, proj_x_v)  # 以视频为k v，音频为q，属于音频分支
# #
# #         # 对于Transformer的输出，恢复其形状
# #         h_av = h_av.permute(0, 2, 1)
# #         h_va = h_va.permute(0, 2, 1)
# #
# #         # 将Transformer的输出添加到分支上
# #         x_audio = h_av + x_audio
# #         x_visual = h_va + x_visual
# #
# #         # 将音频与图像特征，通过后面2层1D卷积
# #         x_audio = self.audio_model.forward_stage2(x_audio)
# #         x_visual = self.visual_model.forward_stage2(x_visual)
# #
# #         # 进行池化操作
# #         audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
# #         video_pooled = x_visual.mean([-1])
# #
# #         # 将池化后的音频特征与图像特征拼接
# #         x = torch.cat((audio_pooled, video_pooled), dim=-1)
# #
# #         # 将拼接特征送入分类器中，进行情绪预测
# #         x1 = self.classifier_1(x)
# #         return x1
# #
# #     # 这个是中期注意机制的前向传播
# #     # def forward_feature_2(self, x_audio, x_visual):
# #     #     x_audio = self.audio_model.forward_stage1(x_audio) # 将音频特征输入2层1D卷积中，得到128维输出
# #     #     x_visual = self.visual_model.forward_features(x_visual) # 将图像通过EfficientFace进行特征提取
# #     #     x_visual = self.visual_model.forward_stage1(x_visual) # 将提取图像特征送入2层1D卷积中得到64维输出
# #     #     print("stage1 audio = ", x_audio.shape)             # ([40, 128, 150])
# #     #     print("stage1 visual =", x_visual.shape)            # ([40, 64, 15])
# #     #     proj_x_a = x_audio.permute(0,2,1) # 调整矩阵行与列
# #     #     proj_x_v = x_visual.permute(0,2,1) # 调整矩阵行与列
# #     #     print("changeshape audio = ", proj_x_a.shape)       # ([40, 150, 128])
# #     #     print("changeshape visual =", proj_x_v.shape)       # ([40, 15, 64])
# #     #     _, h_av = self.av1(proj_x_v, proj_x_a) # 使用注意机制，属于音频分支
# #     #     _, h_va = self.va1(proj_x_a, proj_x_v) # 使用注意机制，属于视频分支
# #     #
# #     #     if h_av.size(1) > 1: #if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
# #     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# #     #
# #     #     h_av = h_av.sum([-2]) # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
# #     #                           #                                                     [3 4 4 ]
# #     #                           # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
# #     #
# #     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
# #     #     if h_va.size(1) > 1: #if more than 1 head, take average
# #     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# #     #
# #     #     h_va = h_va.sum([-2])
# #     #
# #     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
# #     #     x_audio = h_va*x_audio
# #     #     x_visual = h_av*x_visual
# #     #     print("afterattention audio = ", x_audio.shape)     # ([40, 128, 150])
# #     #     print("afterattention visual =", x_visual.shape)    # ([40, 64, 15])
# #     #     # 然后经过后面的2层1D卷积
# #     #     x_audio = self.audio_model.forward_stage2(x_audio)
# #     #     x_visual = self.visual_model.forward_stage2(x_visual)
# #     #     print("stage2 audio = ", x_audio.shape)             # ([40, 128, 144])
# #     #     print("stage2 visual =", x_visual.shape)            # ([40, 128, 15])
# #     #     # 池化层
# #     #     audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
# #     #     video_pooled = x_visual.mean([-1])
# #     #     print("pooled audio = ", audio_pooled.shape)        # ([40, 128])
# #     #     print("pooled visual =", video_pooled.shape)        # ([40, 128])
# #     #
# #     #     # 拼接特征
# #     #     x = torch.cat((audio_pooled, video_pooled), dim=-1)
# #     #
# #     #     # 进行分类
# #     #     x1 = self.classifier_1(x)
# #     #     return x1
# #
# #     def forward_feature_2(self, x_audio, x_visual):
# #         x_audio = self.audio_model.forward_stage1(x_audio)
# #         x_visual = self.visual_model.forward_features(x_visual)
# #         x_visual = self.visual_model.forward_stage1(x_visual)
# #
# #         proj_x_a = x_audio.permute(0, 2, 1)
# #         proj_x_v = x_visual.permute(0, 2, 1)
# #
# #         _, h_av = self.av1(proj_x_v, proj_x_a)
# #         _, h_va = self.va1(proj_x_a, proj_x_v)
# #
# #         if h_av.size(1) > 1:  # if more than 1 head, take average
# #             h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# #
# #         h_av = h_av.sum([-2])
# #
# #         if h_va.size(1) > 1:  # if more than 1 head, take average
# #             h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# #
# #         h_va = h_va.sum([-2])
# #
# #         x_audio = h_va * x_audio
# #         x_visual = h_av * x_visual
# #
# #         x_audio = self.audio_model.forward_stage2(x_audio)
# #         x_visual = self.visual_model.forward_stage2(x_visual)
# #
# #         audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
# #         video_pooled = x_visual.mean([-1])
# #
# #         x = torch.cat((audio_pooled, video_pooled), dim=-1)
# #
# #         x1 = self.classifier_1(x)
# #         return x1
# #
# #     # 这个是后期Transformer的前向传播
# #     def forward_transformer(self, x_audio, x_visual):
# #         print("进入后期Transformer\n")
# #         x_audio = self.audio_model.forward_stage1(x_audio)
# #         print("x_audio.shape stage1 = ", x_audio.shape)  # ([40, 128, 150]) 音频1
# #         proj_x_a = self.audio_model.forward_stage2(x_audio)  # 得到128维输出
# #         print("x_audio.shape stage2 = ", proj_x_a.shape)  # ([40, 128, 144]) 音频2
# #
# #         x_visual = self.visual_model.forward_features(x_visual)
# #         x_visual = self.visual_model.forward_stage1(x_visual)
# #         print("x_visual.shape stage1 = ", x_visual.shape)  # ([40, 64, 15]) 视频1
# #         proj_x_v = self.visual_model.forward_stage2(x_visual)  # 得到128维输出
# #         print("x_visual.shape stage2 = ", proj_x_v.shape)  # ([40, 128, 15]) 视频2
# #
# #         proj_x_a = proj_x_a.permute(0, 2, 1)
# #         proj_x_v = proj_x_v.permute(0, 2, 1)
# #         print("转换形状之后\n")
# #         print("x_audio.shape shape = ", proj_x_a.shape)  # ([40, 144, 128]) 音频
# #         print("x_visual.shape shape = ", proj_x_v.shape)  # ([40, 15, 128]) 视频
# #         h_av = self.av(proj_x_v, proj_x_a)  # Transformer
# #         h_va = self.va(proj_x_a, proj_x_v)
# #         print("经过Transformer后\n")
# #         print("h_av.shape = ", h_av.shape)  # ([40, 144, 128]) 音频
# #         print("h_va.shape = ", h_av.shape)  # ([40, 144, 128]) 视频
# #         # 池化层
# #         audio_pooled = h_av.mean([1])  # mean accross temporal dimension
# #         video_pooled = h_va.mean([1])
# #         print("池化后\n")
# #         print("audio_pooled.shape = ", audio_pooled.shape)  # ([40, 128]) 音频
# #         print("video_pooled.shape = ", video_pooled.shape)  # ([40, 128]) 视频
# #         # 拼接特征
# #         x = torch.cat((audio_pooled, video_pooled), dim=-1)
# #         print("拼接后\n")
# #         print("x.shape = ", x.shape)  # ([40, 256])
# #
# #         # 情感预测
# #         x1 = self.classifier_1(x)
# #         print("预测后\n")
# #         print("x1.shape= ", x1.shape)  # ([40, 8])
# #         return x1
# #
# #     # 自己改的，想要Attention中期 单头注意机制 + Transformer后期 + 两个单模态，最后改下线性层的输入维度为e_dim * 4就好 然后把4个特征拼接起来
# #     # 但是希望在两个单模态的分支上，能够添加损失函数与中间的分支做交互，因为目的是为了补偿
# #     # 下面这个是没有辅助网络的，主干网络只有Attention和Transformer
# #     # def forward_own(self, x_audio, x_visual):
# #     #     x_audio_aux = x_audio # 音频辅助网络
# #     #
# #     #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
# #     #     # print("音频 stage1 = ", x_audio.shape)
# #     #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
# #     #
# #     #     x_visual_aux = x_visual # 图像辅助网络
# #     #
# #     #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
# #     #     # print("视频 stage1 = ", x_visual.shape)
# #     #
# #     #     # 辅助网络
# #     #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
# #     #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
# #     #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
# #     #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
# #     #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
# #     #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
# #     #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
# #     #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
# #     #
# #     #     # 辅助网池化层
# #     #     audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
# #     #     visual_aux_pooled = x_visual_aux.mean([-1])
# #     #     # # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)
# #     #     # # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)
# #     #
# #     #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
# #     #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
# #     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
# #     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
# #     #
# #     #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
# #     #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
# #     #
# #     #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
# #     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# #     #
# #     #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
# #     #     #                                                     [3 4 4 ]
# #     #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
# #     #
# #     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
# #     #     if h_va.size(1) > 1:  # if more than 1 head, take average
# #     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# #     #
# #     #     h_va = h_va.sum([-2])
# #     #
# #     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
# #     #     x_audio = h_va * x_audio
# #     #     x_visual = h_av * x_visual
# #     #     # print("音频 注意机制 x_audio = ", x_audio.shape)
# #     #     # print("图像 注意机制 x_visual = ", x_visual.shape)
# #     #
# #     #     # 然后经过后面的2层1D卷积
# #     #     x_audio = self.audio_model.forward_stage2(x_audio)
# #     #     x_visual = self.visual_model.forward_stage2(x_visual)
# #     #     # print("音频 stage2 x_audio = ", x_audio.shape)
# #     #     # print("图像 stage2 x_visual = ", x_visual.shape)
# #     #
# #     #     # 使用后期Transformer
# #     #     proj_x_a = x_audio.permute(0, 2, 1)
# #     #     proj_x_v = x_visual.permute(0, 2, 1)
# #     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
# #     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
# #     #     h_av = self.av(proj_x_v, proj_x_a) # Transformer
# #     #     h_va = self.va(proj_x_a, proj_x_v)
# #     #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
# #     #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
# #     #
# #     #     # 池化层
# #     #     audio_pooled = h_av.mean([1])  # mean accross temporal dimension
# #     #     video_pooled = h_va.mean([1])
# #     #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
# #     #     # print("视频 池化 video_pooled = ", video_pooled.shape)
# #     #
# #     #     # 拼接特征
# #     #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
# #     #     x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
# #     #
# #     #     # 进行分类
# #     #     x1 = self.classifier_1(x)
# #     #     return x1
# #
# #     # # 辅助网络有包含中期Transformer块
# #     # def forward_own(self, x_audio, x_visual):
# #     #     x_audio_aux = x_audio # 音频辅助网络
# #     #
# #     #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
# #     #     # print("音频 stage1 = ", x_audio.shape)
# #     #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
# #     #
# #     #     x_visual_aux = x_visual # 图像辅助网络
# #     #
# #     #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
# #     #     # print("视频 stage1 = ", x_visual.shape)
# #     #
# #     #     # 辅助网络1阶段
# #     #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
# #     #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
# #     #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
# #     #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
# #     #
# #     #
# #     #     # # 辅助网络中期Attention
# #     #     # proj_x_aa = x_audio_aux.permute(0, 2, 1)
# #     #     # proj_x_vv = x_visual_aux.permute(0, 2, 1)
# #     #     # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
# #     #     # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
# #     #     # if h_aa.size(1) > 1:
# #     #     #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
# #     #     # h_aa = h_aa.sum([-2])
# #     #     # if h_vv.size(1) > 1:
# #     #     #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
# #     #     # h_vv = h_vv.sum([-2])
# #     #     # x_audio_aux = h_aa * x_audio_aux
# #     #     # x_visual_aux = h_vv * x_visual_aux
# #     #
# #     #     # 辅助网络2阶段
# #     #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
# #     #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
# #     #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
# #     #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
# #     #
# #     #     # 辅助网络转换
# #     #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
# #     #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
# #     #     # print("转换形状之后\n")
# #     #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
# #     #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
# #     #
# #     #     # 辅助网络后期自注意
# #     #     x_audio_aux = self.av(x_audio_aux, x_audio_aux) # Transformer
# #     #     x_visual_aux = self.va(x_visual_aux, x_visual_aux)
# #     #     # print("经过Transformer后\n")
# #     #     # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
# #     #     # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
# #     #
# #     #     # 辅助网络转换
# #     #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
# #     #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
# #     #     # print("转换形状之后\n")
# #     #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
# #     #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
# #     #
# #     #     # 辅助网池化层
# #     #     audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
# #     #     visual_aux_pooled = x_visual_aux.mean([-1])
# #     #     # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
# #     #     # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
# #     #
# #     #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
# #     #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
# #     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
# #     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
# #     #
# #     #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
# #     #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
# #     #
# #     #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
# #     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# #     #
# #     #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
# #     #     #                                                     [3 4 4 ]
# #     #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
# #     #
# #     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
# #     #     if h_va.size(1) > 1:  # if more than 1 head, take average
# #     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# #     #
# #     #     h_va = h_va.sum([-2])
# #     #
# #     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
# #     #     x_audio = h_va * x_audio
# #     #     x_visual = h_av * x_visual
# #     #     # print("音频 注意机制 x_audio = ", x_audio.shape)
# #     #     # print("图像 注意机制 x_visual = ", x_visual.shape)
# #     #
# #     #     # 然后经过后面的2层1D卷积
# #     #     x_audio = self.audio_model.forward_stage2(x_audio)
# #     #     x_visual = self.visual_model.forward_stage2(x_visual)
# #     #     # print("音频 stage2 x_audio = ", x_audio.shape)
# #     #     # print("图像 stage2 x_visual = ", x_visual.shape)
# #     #
# #     #     # 使用后期Transformer
# #     #     proj_x_a = x_audio.permute(0, 2, 1)
# #     #     proj_x_v = x_visual.permute(0, 2, 1)
# #     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
# #     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
# #     #     h_av = self.av(proj_x_v, proj_x_a) # Transformer
# #     #     h_va = self.va(proj_x_a, proj_x_v)
# #     #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
# #     #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
# #     #
# #     #     # 池化层
# #     #     audio_pooled = h_av.mean([1])  # mean accross temporal dimension
# #     #     video_pooled = h_va.mean([1])
# #     #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
# #     #     # print("视频 池化 video_pooled = ", video_pooled.shape)
# #     #
# #     #     # 拼接特征
# #     #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
# #     #     x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
# #     #
# #     #     # 进行分类
# #     #     x1 = self.classifier_1(x)
# #     #     return x1
# #     #
# #
# #     # 辅助网络有包含中期Transformer块 添加co-attention
# #     def forward_own(self, x_audio, x_visual):
# #         x_audio_aux = x_audio  # 音频辅助网络
# #
# #         x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
# #         # print("音频 stage1 = ", x_audio.shape)
# #         x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
# #
# #         x_visual_aux = x_visual  # 图像辅助网络
# #
# #         x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
# #         # print("视频 stage1 = ", x_visual.shape)
# #
# #         # 辅助网络1阶段
# #         x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
# #         # print("辅助音频 stage1 = ", x_audio_aux.shape)
# #         x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
# #         # print("辅助视频 stage1 = ", x_visual_aux.shape)
# #
# #         # # # 辅助网络中期Attention
# #         # proj_x_aa = x_audio_aux.permute(0, 2, 1)
# #         # proj_x_vv = x_visual_aux.permute(0, 2, 1)
# #         # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
# #         # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
# #         # if h_aa.size(1) > 1:
# #         #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
# #         # h_aa = h_aa.sum([-2])
# #         # if h_vv.size(1) > 1:
# #         #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
# #         # h_vv = h_vv.sum([-2])
# #         # x_audio_aux = h_aa * x_audio_aux
# #         # x_visual_aux = h_vv * x_visual_aux
# #
# #         # 辅助网络2阶段
# #         x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
# #         # print("辅助音频 stage2 = ", x_audio_aux.shape)
# #         x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
# #         # print("辅助视频 stage2 = ", x_visual_aux.shape)
# #
# #         # 辅助网络转换
# #         x_audio_aux = x_audio_aux.permute(0, 2, 1)
# #         x_visual_aux = x_visual_aux.permute(0, 2, 1)
# #         # print("转换形状之后\n")
# #         # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
# #         # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
# #
# #         # 辅助网络后期自注意
# #         x_audio_aux = self.av(x_audio_aux, x_audio_aux)  # Transformer
# #         x_visual_aux = self.va(x_visual_aux, x_visual_aux)
# #         # print("经过Transformer后\n")
# #         # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
# #         # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
# #
# #         # 辅助网络转换
# #         x_audio_aux = x_audio_aux.permute(0, 2, 1)
# #         x_visual_aux = x_visual_aux.permute(0, 2, 1)
# #         # print("转换形状之后\n")
# #         # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
# #         # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
# #
# #         # 辅助网池化层
# #         audio_aux_pooled = x_audio_aux.mean([-1])  # mean accross temporal dimension
# #         visual_aux_pooled = x_visual_aux.mean([-1])
# #         # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
# #         # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
# #
# #         proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
# #         proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
# #         # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
# #         # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
# #
# #         _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
# #         _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
# #
# #         if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
# #             h_av = torch.mean(h_av, axis=1).unsqueeze(1)
# #
# #         h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
# #         #                                                     [3 4 4 ]
# #         # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
# #
# #         # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
# #         if h_va.size(1) > 1:  # if more than 1 head, take average
# #             h_va = torch.mean(h_va, axis=1).unsqueeze(1)
# #
# #         h_va = h_va.sum([-2])
# #
# #         # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
# #         x_audio = h_va * x_audio
# #         x_visual = h_av * x_visual
# #         # print("音频 注意机制 x_audio = ", x_audio.shape)
# #         # print("图像 注意机制 x_visual = ", x_visual.shape)
# #
# #         # 然后经过后面的2层1D卷积
# #         x_audio = self.audio_model.forward_stage2(x_audio)
# #         x_visual = self.visual_model.forward_stage2(x_visual)
# #         # print("音频 stage2 x_audio = ", x_audio.shape)
# #         # print("图像 stage2 x_visual = ", x_visual.shape)
# #
# #         # 使用后期Transformer
# #         proj_x_a = x_audio.permute(0, 2, 1)
# #         proj_x_v = x_visual.permute(0, 2, 1)
# #         # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
# #         # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
# #         h_av = self.av(proj_x_v, proj_x_a)  # Transformer
# #         h_va = self.va(proj_x_a, proj_x_v)
# #         # print("音频 Transformer之后 x_audio = ", h_av.shape)
# #         # print("图像 Transformer之后 x_visual = ", h_va.shape)
# #
# #         # 主干网络使用co-attention
# #         h_co = self.co_av(h_av, h_va)
# #         # 池化层
# #         h_co_pool = h_co.mean([1])
# #         # audio_pooled = h_av.mean([1])  # mean accross temporal dimension
# #         # video_pooled = h_va.mean([1])
# #         # print("音频 池化 audio_pooled = ", audio_pooled.shape)
# #         # print("视频 池化 video_pooled = ", video_pooled.shape)
# #
# #         # 拼接特征
# #         # x = torch.cat((audio_pooled, video_pooled), dim=-1)
# #         # x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
# #         x = torch.cat((audio_aux_pooled, h_co_pool, visual_aux_pooled), dim=-1)
# #
# #         # 进行分类
# #         x1 = self.classifier_1(x)
# #         return x1
# #
# #
# #
# #
# # -*- coding: utf-8 -*-
# """
# Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
# """
#
# import torch
# import torch.nn as nn
# from models.modulator import Modulator
# from models.efficientface import LocalFeatureExtractor, InvertedResidual
# from models.transformer_timm import AttentionBlock, Attention
# from models.co_attention import CoAttention
#
#
# def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
#     return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#                          nn.BatchNorm1d(out_channels),
#                          nn.ReLU(inplace=True))
#
#
# class EfficientFaceTemporal(nn.Module):  # EfficientFaceTemporal模型 这个模型是21年发表在AAI上的一个轻量级面部表情识别模型
#
#     def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
#         super(EfficientFaceTemporal, self).__init__()
#
#         if len(stages_repeats) != 3:
#             raise ValueError('expected stages_repeats as list of 3 positive ints')
#         if len(stages_out_channels) != 5:
#             raise ValueError('expected stages_out_channels as list of 5 positive ints')
#         self._stage_out_channels = stages_out_channels
#
#         # 这段代码定义了模型的一个卷积层，它包含一个2D卷积操作，一个批量化诡异操作，以及一个ReLu激活函数
#         input_channels = 3
#         output_channels = self._stage_out_channels[0]
#         self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
#                                    nn.BatchNorm2d(output_channels),
#                                    nn.ReLU(inplace=True), )
#
#         # 这段代码更新通道数量，便于下一层处理
#         input_channels = output_channels
#
#         # 定义最大池化
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # 定义模型三个结点，每个阶段包含一系列的InvertedResidual块
#         stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
#         for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
#             seq = [InvertedResidual(input_channels, output_channels, 2)]
#             for i in range(repeats - 1):
#                 seq.append(InvertedResidual(output_channels, output_channels, 1))
#             setattr(self, name, nn.Sequential(*seq))
#             input_channels = output_channels
#
#         # 局部特征提取器与调制器
#         self.local = LocalFeatureExtractor(29, 116, 1)
#         self.modulator = Modulator(116)
#
#         # 这几行代码定义了模型的第五个卷积层，包含一个2D卷积操作，一个批量归一化操作，以及一个ReLU激活函数
#         output_channels = self._stage_out_channels[-1]
#
#         self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
#                                    nn.BatchNorm2d(output_channels),
#                                    nn.ReLU(inplace=True), )
#
#         # 四行代码定义了四个1D卷积快
#         self.conv1d_0 = conv1d_block(output_channels, 64)
#         self.conv1d_1 = conv1d_block(64, 64)
#         self.conv1d_2 = conv1d_block(64, 128)
#         self.conv1d_3 = conv1d_block(128, 128)
#
#         # 图像辅助网络
#         # 四行代码定义了四个1D卷积快
#         self.conv1d_0_viusal_aux = conv1d_block(output_channels, 64)
#         self.conv1d_1_visual_aux = conv1d_block(64, 64)
#         self.conv1d_2_visual_aux = conv1d_block(64, 128)
#         self.conv1d_3_visual_aux = conv1d_block(128, 128)
#
#         # 线性分类器
#         self.classifier_1 = nn.Sequential(
#             nn.Linear(128, num_classes),
#         )
#
#         # 将每个样本图像保存为类的一个属性
#         self.im_per_sample = im_per_sample
#
#     def forward_features(self, x):
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.modulator(self.stage2(x)) + self.local(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         x = self.conv5(x)
#         x = x.mean([2, 3])  # global average pooling
#         return x
#
#     # 对输入的视频x应用一些列的卷积，池化，调制器，以及局部特征提取器，然后对结果进行平均池化，并返回池化后结果 对应于图中EfficientFace提取图像特征
#
#     def forward_stage1(self, x):
#         # Getting samples per batch
#         assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
#         n_samples = x.shape[0] // self.im_per_sample
#         x = x.view(n_samples, self.im_per_sample, x.shape[1])
#         x = x.permute(0, 2, 1)
#         x = self.conv1d_0(x)
#         x = self.conv1d_1(x)
#         return x
#
#     # 检查x输入的形状是否满足预期，然后对x进行重塑与置换，再对x应用两个1D卷积快，并返回结果 对应图中图像分支前2个1D卷积
#
#     # 图像辅助网络
#     def forward_stage1_visual_aux(self, x):
#         # Getting samples per batch
#         assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
#         n_samples = x.shape[0] // self.im_per_sample
#         x = x.view(n_samples, self.im_per_sample, x.shape[1])
#         x = x.permute(0, 2, 1)
#         x = self.conv1d_0_viusal_aux(x)
#         x = self.conv1d_1_visual_aux(x)
#         return x
#
#     def forward_stage2(self, x):
#         x = self.conv1d_2(x)
#         x = self.conv1d_3(x)
#         return x
#
#     # 对输入x应用两个1D卷积块并返回结果 对应图中图像分支后2个1D卷积
#
#     # 图像辅助网络
#     def forward_stage2_visual_aux(self, x):
#         x = self.conv1d_2_visual_aux(x)
#         x = self.conv1d_3_visual_aux(x)
#         return x
#
#     def forward_classifier(self, x):
#         x = x.mean([-1])  # pooling accross temporal dimension
#         x1 = self.classifier_1(x)
#         return x1
#
#     # 对输入x进行平均池化，减少维度，然后将池化后x输入分类器，并返回分类器输出
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.forward_stage1(x)
#         x = self.forward_stage2(x)
#         x = self.forward_classifier(x)
#         return x
#     # 对输入的x先进性forward_features进行特征提取，然后依次执行forward_stage1，forward_stage2和forward_classifier
#     # 这个就类比图中整个图像分支 （注意哈，相当于单模态的图像情感预测，没有注意机制，就是先调用forward_features提取图像特征，然后将特征通过
#     # forward_stage1和forward_stage2，对应图中4个1D卷积，最后通过forward_classifier线性层输出预测情感）
#
#
# # 接收两个参数，参数一模型，参数二路径 这里的路径是训练好模型的路径 实现的功能就是把训练好的模型参数加载进模型中
# def init_feature_extractor(model, path):
#     if path == 'None' or path is None:
#         return
#     checkpoint = torch.load(path, map_location=torch.device('cpu'))  # 从path路径加载检查点
#     pre_trained_dict = checkpoint['state_dict']  # 从检查点中获取状态
#     pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
#     print('Initializing efficientnet')  # 打印提示信息，表示初始化EfficientNet模型
#     model.load_state_dict(pre_trained_dict, strict=False)  # 将预训练模型加载值model中
#
#
# # 参数一，类别数，参数二，人物，参数三，序列长度 创建一个EfficientFaceTemporal模型，并返回
# def get_model(num_classes, task, seq_length):
#     model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
#     return model
#
#
# # 参数一，输入通道数，参数二，输出通道数，参数三，卷积核大小，参数四，补偿，参数五，填充方法
# # 创建一个1D卷积块，包含一1D卷积层，一批量归一化层，一ReLU激活函数，一最大池化层
# def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
#     return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
#                          nn.BatchNorm1d(out_channels),
#                          nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))
#
#
# class AudioCNNPool(nn.Module):  # AudioCNNPool模型实例，用于处理音频数据
#
#     def __init__(self, num_classes=8):
#         super(AudioCNNPool, self).__init__()
#
#         # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
#         input_channels = 10
#         self.conv1d_0 = conv1d_block_audio(input_channels, 64)
#         self.conv1d_1 = conv1d_block_audio(64, 128)
#         self.conv1d_2 = conv1d_block_audio(128, 256)
#         self.conv1d_3 = conv1d_block_audio(256, 128)
#
#         # 音频辅助网络
#         # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
#         input_channels = 10
#         self.conv1d_0_audio_aux = conv1d_block_audio(input_channels, 64)
#         self.conv1d_1_audio_aux = conv1d_block_audio(64, 128)
#         self.conv1d_2_audio_aux = conv1d_block_audio(128, 256)
#         self.conv1d_3_audio_aux = conv1d_block_audio(256, 128)
#
#         # 定义一个线性分类层
#         self.classifier_1 = nn.Sequential(
#             nn.Linear(128, num_classes),
#         )
#
#     # 四个卷积，然后分类器，输出音频情感预测
#     def forward(self, x):
#         x = self.forward_stage1(x)
#         x = self.forward_stage2(x)
#         x = self.forward_classifier(x)
#         return x
#
#     def forward_stage1(self, x):
#         x = self.conv1d_0(x)
#         x = self.conv1d_1(x)
#         return x
#
#     # 音频辅助网络
#     def forward_stage1_audio_aux(self, x):
#         x = self.conv1d_0_audio_aux(x)
#         x = self.conv1d_1_audio_aux(x)
#         return x
#
#     def forward_stage2(self, x):
#         x = self.conv1d_2(x)
#         x = self.conv1d_3(x)
#         return x
#
#     # 音频辅助网络
#     def forward_stage2_audio_aux(self, x):
#         x = self.conv1d_2_audio_aux(x)
#         x = self.conv1d_3_audio_aux(x)
#         return x
#
#     def forward_classifier(self, x):
#         x = x.mean([-1])  # pooling accross temporal dimension
#         x1 = self.classifier_1(x)
#         return x1
#
#
# class MultiModalCNN(nn.Module):
#     def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None',
#                  num_heads=1):  # 这里的pretr_ef就是预训练EfficientFace_Trained_on_AffectNet7.pth.tar
#         super(MultiModalCNN, self).__init__()
#         assert fusion in ['ia', 'it', 'lt'], print(
#             'Unsupported fusion method: {}'.format(fusion))  # 这行代码检查fusion参数是否在['ia', 'it', 'lt']中。如果不在，就会打印一条错误消息。
#
#         self.audio_model = AudioCNNPool(num_classes=num_classes)  # 这行代码创建了一个AudioCNNPool模型实例，用于处理音频数据。
#         self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes,
#                                                   seq_length)  # 这行代码创建了一个EfficientFaceTemporal模型实例，用于处理视觉数据
#
#         init_feature_extractor(self.visual_model, pretr_ef)  # 这行代码使用预训练的模型初始化EfficientFaceTemporal模型
#
#         e_dim = 128  # 嵌入维度
#         input_dim_video = 128  # 视频输入维度
#         input_dim_audio = 128  # 音频的输入维度
#         self.fusion = fusion  # 融合方法
#
#         # 通过不同的融合方法，创建对应注意力块
#         # Transformer对应为AttentionBlock，而Attention机制对应为Attention()
#         # 这里的lt就表示后期融合Transformer it表示中期融合Transformer
#         # if fusion in ['lt', 'it']:
#         #     if fusion  == 'lt':
#         #         self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
#         #         self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
#         #     elif fusion == 'it':
#         #         input_dim_video = input_dim_video // 2
#         #         self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
#         #         self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
#         # # 这里ia表示中期融合的注意机制
#         # elif fusion in ['ia']:
#         #     input_dim_video = input_dim_video // 2
#         #
#         #     self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
#         #     self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
#
#         # 后期Transformer
#         self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
#                                  num_heads=num_heads)
#         self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
#                                  num_heads=num_heads)
#
#         # 辅助网络后期Transfromer
#         self.aa1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_audio, out_dim=e_dim,
#                                   num_heads=num_heads)
#         self.vv1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_video, out_dim=e_dim,
#                                   num_heads=num_heads)
#
#         # 中期Attention
#         input_dim_video = input_dim_video // 2
#         self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
#                              num_heads=num_heads)
#         self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
#                              num_heads=num_heads)
#
#         # # 辅助网络Attention
#         self.aa = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
#                             num_heads=num_heads)
#         self.vv = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_video, out_dim=input_dim_video,
#                             num_heads=num_heads)
#
#         # 主干网络的co-attention机制
#         self.co_av = CoAttention(input_dim=128, hidden_dim=64)
#
#         # 主干辅助aa的co-attention机制
#         self.co_aa = CoAttention(input_dim=128, hidden_dim=64)
#
#         # 主干辅助vv的co-attention机制
#         self.co_vv = CoAttention(input_dim=128, hidden_dim=64)
#
#         # self.classifier_1 = nn.Sequential( # 创建一个线性分类器
#         #             nn.Linear(e_dim*2, num_classes), # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
#         #         )
#
#         self.classifier_1 = nn.Sequential(  # 创建一个线性分类器
#             nn.Linear(e_dim * 3, num_classes),  # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
#         )
#
#     def forward(self, x_audio, x_visual):  # 是MultiModalCNN类的主要方法，它通过融合方法fusion来决定使用哪一个前线传播
#
#         return self.forward_own(x_audio, x_visual)
#         # if self.fusion == 'lt':
#         #     return self.forward_transformer(x_audio, x_visual)
#         #
#         # elif self.fusion == 'ia':
#         #     return self.forward_feature_2(x_audio, x_visual)
#         #
#         # elif self.fusion == 'it':
#         #     return self.forward_feature_3(x_audio, x_visual)
#
#     # PS：输入的音频特征与图像特征都是128维度
#
#     # 这个是中期Transformer的前向传播
#     def forward_feature_3(self, x_audio, x_visual):
#         x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频数据输入2个1D卷积中，得到128维度输出
#         x_visual = self.visual_model.forward_features(x_visual)  # 将图像数据输入特征提取模块
#         x_visual = self.visual_model.forward_stage1(x_visual)  # 在经过2个1D卷积，得到64维输出
#
#         # 假设有一个3维张量，形状是(2, 3, 4)，其中的2表示由2个矩阵，每个矩阵有3行4列，如果使用permute(0, 2, 1)后，那么张量变为(2, 4, 3)
#         proj_x_a = x_audio.permute(0, 2, 1)
#         proj_x_v = x_visual.permute(0, 2, 1)
#
#         # 这两行使用Transformer机制
#         h_av = self.av1(proj_x_v, proj_x_a)  # 以音频为k v，视频为q，属于视频分支
#         h_va = self.va1(proj_x_a, proj_x_v)  # 以视频为k v，音频为q，属于音频分支
#
#         # 对于Transformer的输出，恢复其形状
#         h_av = h_av.permute(0, 2, 1)
#         h_va = h_va.permute(0, 2, 1)
#
#         # 将Transformer的输出添加到分支上
#         x_audio = h_av + x_audio
#         x_visual = h_va + x_visual
#
#         # 将音频与图像特征，通过后面2层1D卷积
#         x_audio = self.audio_model.forward_stage2(x_audio)
#         x_visual = self.visual_model.forward_stage2(x_visual)
#
#         # 进行池化操作
#         audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
#         video_pooled = x_visual.mean([-1])
#
#         # 将池化后的音频特征与图像特征拼接
#         x = torch.cat((audio_pooled, video_pooled), dim=-1)
#
#         # 将拼接特征送入分类器中，进行情绪预测
#         x1 = self.classifier_1(x)
#         return x1
#
#     # 这个是中期注意机制的前向传播
#     # def forward_feature_2(self, x_audio, x_visual):
#     #     x_audio = self.audio_model.forward_stage1(x_audio) # 将音频特征输入2层1D卷积中，得到128维输出
#     #     x_visual = self.visual_model.forward_features(x_visual) # 将图像通过EfficientFace进行特征提取
#     #     x_visual = self.visual_model.forward_stage1(x_visual) # 将提取图像特征送入2层1D卷积中得到64维输出
#     #     print("stage1 audio = ", x_audio.shape)             # ([40, 128, 150])
#     #     print("stage1 visual =", x_visual.shape)            # ([40, 64, 15])
#     #     proj_x_a = x_audio.permute(0,2,1) # 调整矩阵行与列
#     #     proj_x_v = x_visual.permute(0,2,1) # 调整矩阵行与列
#     #     print("changeshape audio = ", proj_x_a.shape)       # ([40, 150, 128])
#     #     print("changeshape visual =", proj_x_v.shape)       # ([40, 15, 64])
#     #     _, h_av = self.av1(proj_x_v, proj_x_a) # 使用注意机制，属于音频分支
#     #     _, h_va = self.va1(proj_x_a, proj_x_v) # 使用注意机制，属于视频分支
#     #
#     #     if h_av.size(1) > 1: #if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
#     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
#     #
#     #     h_av = h_av.sum([-2]) # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
#     #                           #                                                     [3 4 4 ]
#     #                           # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
#     #
#     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
#     #     if h_va.size(1) > 1: #if more than 1 head, take average
#     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
#     #
#     #     h_va = h_va.sum([-2])
#     #
#     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
#     #     x_audio = h_va*x_audio
#     #     x_visual = h_av*x_visual
#     #     print("afterattention audio = ", x_audio.shape)     # ([40, 128, 150])
#     #     print("afterattention visual =", x_visual.shape)    # ([40, 64, 15])
#     #     # 然后经过后面的2层1D卷积
#     #     x_audio = self.audio_model.forward_stage2(x_audio)
#     #     x_visual = self.visual_model.forward_stage2(x_visual)
#     #     print("stage2 audio = ", x_audio.shape)             # ([40, 128, 144])
#     #     print("stage2 visual =", x_visual.shape)            # ([40, 128, 15])
#     #     # 池化层
#     #     audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
#     #     video_pooled = x_visual.mean([-1])
#     #     print("pooled audio = ", audio_pooled.shape)        # ([40, 128])
#     #     print("pooled visual =", video_pooled.shape)        # ([40, 128])
#     #
#     #     # 拼接特征
#     #     x = torch.cat((audio_pooled, video_pooled), dim=-1)
#     #
#     #     # 进行分类
#     #     x1 = self.classifier_1(x)
#     #     return x1
#
#     def forward_feature_2(self, x_audio, x_visual):
#         x_audio = self.audio_model.forward_stage1(x_audio)
#         x_visual = self.visual_model.forward_features(x_visual)
#         x_visual = self.visual_model.forward_stage1(x_visual)
#
#         proj_x_a = x_audio.permute(0, 2, 1)
#         proj_x_v = x_visual.permute(0, 2, 1)
#
#         _, h_av = self.av1(proj_x_v, proj_x_a)
#         _, h_va = self.va1(proj_x_a, proj_x_v)
#
#         if h_av.size(1) > 1:  # if more than 1 head, take average
#             h_av = torch.mean(h_av, axis=1).unsqueeze(1)
#
#         h_av = h_av.sum([-2])
#
#         if h_va.size(1) > 1:  # if more than 1 head, take average
#             h_va = torch.mean(h_va, axis=1).unsqueeze(1)
#
#         h_va = h_va.sum([-2])
#
#         x_audio = h_va * x_audio
#         x_visual = h_av * x_visual
#
#         x_audio = self.audio_model.forward_stage2(x_audio)
#         x_visual = self.visual_model.forward_stage2(x_visual)
#
#         audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
#         video_pooled = x_visual.mean([-1])
#
#         x = torch.cat((audio_pooled, video_pooled), dim=-1)
#
#         x1 = self.classifier_1(x)
#         return x1
#
#     # 这个是后期Transformer的前向传播
#     def forward_transformer(self, x_audio, x_visual):
#         print("进入后期Transformer\n")
#         x_audio = self.audio_model.forward_stage1(x_audio)
#         print("x_audio.shape stage1 = ", x_audio.shape)  # ([40, 128, 150]) 音频1
#         proj_x_a = self.audio_model.forward_stage2(x_audio)  # 得到128维输出
#         print("x_audio.shape stage2 = ", proj_x_a.shape)  # ([40, 128, 144]) 音频2
#
#         x_visual = self.visual_model.forward_features(x_visual)
#         x_visual = self.visual_model.forward_stage1(x_visual)
#         print("x_visual.shape stage1 = ", x_visual.shape)  # ([40, 64, 15]) 视频1
#         proj_x_v = self.visual_model.forward_stage2(x_visual)  # 得到128维输出
#         print("x_visual.shape stage2 = ", proj_x_v.shape)  # ([40, 128, 15]) 视频2
#
#         proj_x_a = proj_x_a.permute(0, 2, 1)
#         proj_x_v = proj_x_v.permute(0, 2, 1)
#         print("转换形状之后\n")
#         print("x_audio.shape shape = ", proj_x_a.shape)  # ([40, 144, 128]) 音频
#         print("x_visual.shape shape = ", proj_x_v.shape)  # ([40, 15, 128]) 视频
#         h_av = self.av(proj_x_v, proj_x_a)  # Transformer
#         h_va = self.va(proj_x_a, proj_x_v)
#         print("经过Transformer后\n")
#         print("h_av.shape = ", h_av.shape)  # ([40, 144, 128]) 音频
#         print("h_va.shape = ", h_av.shape)  # ([40, 144, 128]) 视频
#         # 池化层
#         audio_pooled = h_av.mean([1])  # mean accross temporal dimension
#         video_pooled = h_va.mean([1])
#         print("池化后\n")
#         print("audio_pooled.shape = ", audio_pooled.shape)  # ([40, 128]) 音频
#         print("video_pooled.shape = ", video_pooled.shape)  # ([40, 128]) 视频
#         # 拼接特征
#         x = torch.cat((audio_pooled, video_pooled), dim=-1)
#         print("拼接后\n")
#         print("x.shape = ", x.shape)  # ([40, 256])
#
#         # 情感预测
#         x1 = self.classifier_1(x)
#         print("预测后\n")
#         print("x1.shape= ", x1.shape)  # ([40, 8])
#         return x1
#
#     # 自己改的，想要Attention中期 单头注意机制 + Transformer后期 + 两个单模态，最后改下线性层的输入维度为e_dim * 4就好 然后把4个特征拼接起来
#     # 但是希望在两个单模态的分支上，能够添加损失函数与中间的分支做交互，因为目的是为了补偿
#     # 下面这个是没有辅助网络的，主干网络只有Attention和Transformer
#     # def forward_own(self, x_audio, x_visual):
#     #     x_audio_aux = x_audio # 音频辅助网络
#     #
#     #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
#     #     # print("音频 stage1 = ", x_audio.shape)
#     #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
#     #
#     #     x_visual_aux = x_visual # 图像辅助网络
#     #
#     #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
#     #     # print("视频 stage1 = ", x_visual.shape)
#     #
#     #     # 辅助网络
#     #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
#     #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
#     #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
#     #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
#     #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
#     #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
#     #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
#     #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
#     #
#     #     # 辅助网池化层
#     #     audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
#     #     visual_aux_pooled = x_visual_aux.mean([-1])
#     #     # # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)
#     #     # # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)
#     #
#     #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
#     #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
#     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
#     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
#     #
#     #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
#     #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
#     #
#     #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
#     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
#     #
#     #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
#     #     #                                                     [3 4 4 ]
#     #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
#     #
#     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
#     #     if h_va.size(1) > 1:  # if more than 1 head, take average
#     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
#     #
#     #     h_va = h_va.sum([-2])
#     #
#     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
#     #     x_audio = h_va * x_audio
#     #     x_visual = h_av * x_visual
#     #     # print("音频 注意机制 x_audio = ", x_audio.shape)
#     #     # print("图像 注意机制 x_visual = ", x_visual.shape)
#     #
#     #     # 然后经过后面的2层1D卷积
#     #     x_audio = self.audio_model.forward_stage2(x_audio)
#     #     x_visual = self.visual_model.forward_stage2(x_visual)
#     #     # print("音频 stage2 x_audio = ", x_audio.shape)
#     #     # print("图像 stage2 x_visual = ", x_visual.shape)
#     #
#     #     # 使用后期Transformer
#     #     proj_x_a = x_audio.permute(0, 2, 1)
#     #     proj_x_v = x_visual.permute(0, 2, 1)
#     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
#     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
#     #     h_av = self.av(proj_x_v, proj_x_a) # Transformer
#     #     h_va = self.va(proj_x_a, proj_x_v)
#     #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
#     #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
#     #
#     #     # 池化层
#     #     audio_pooled = h_av.mean([1])  # mean accross temporal dimension
#     #     video_pooled = h_va.mean([1])
#     #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
#     #     # print("视频 池化 video_pooled = ", video_pooled.shape)
#     #
#     #     # 拼接特征
#     #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
#     #     x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
#     #
#     #     # 进行分类
#     #     x1 = self.classifier_1(x)
#     #     return x1
#
#     # # 辅助网络有包含中期Transformer块
#     # def forward_own(self, x_audio, x_visual):
#     #     x_audio_aux = x_audio # 音频辅助网络
#     #
#     #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
#     #     # print("音频 stage1 = ", x_audio.shape)
#     #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
#     #
#     #     x_visual_aux = x_visual # 图像辅助网络
#     #
#     #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
#     #     # print("视频 stage1 = ", x_visual.shape)
#     #
#     #     # 辅助网络1阶段
#     #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
#     #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
#     #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
#     #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
#     #
#     #
#     #     # # 辅助网络中期Attention
#     #     # proj_x_aa = x_audio_aux.permute(0, 2, 1)
#     #     # proj_x_vv = x_visual_aux.permute(0, 2, 1)
#     #     # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
#     #     # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
#     #     # if h_aa.size(1) > 1:
#     #     #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
#     #     # h_aa = h_aa.sum([-2])
#     #     # if h_vv.size(1) > 1:
#     #     #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
#     #     # h_vv = h_vv.sum([-2])
#     #     # x_audio_aux = h_aa * x_audio_aux
#     #     # x_visual_aux = h_vv * x_visual_aux
#     #
#     #     # 辅助网络2阶段
#     #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
#     #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
#     #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
#     #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
#     #
#     #     # 辅助网络转换
#     #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
#     #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
#     #     # print("转换形状之后\n")
#     #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
#     #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
#     #
#     #     # 辅助网络后期自注意
#     #     x_audio_aux = self.av(x_audio_aux, x_audio_aux) # Transformer
#     #     x_visual_aux = self.va(x_visual_aux, x_visual_aux)
#     #     # print("经过Transformer后\n")
#     #     # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
#     #     # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
#     #
#     #     # 辅助网络转换
#     #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
#     #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
#     #     # print("转换形状之后\n")
#     #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
#     #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
#     #
#     #     # 辅助网池化层
#     #     audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
#     #     visual_aux_pooled = x_visual_aux.mean([-1])
#     #     # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
#     #     # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
#     #
#     #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
#     #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
#     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
#     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
#     #
#     #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
#     #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
#     #
#     #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
#     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
#     #
#     #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
#     #     #                                                     [3 4 4 ]
#     #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
#     #
#     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
#     #     if h_va.size(1) > 1:  # if more than 1 head, take average
#     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
#     #
#     #     h_va = h_va.sum([-2])
#     #
#     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
#     #     x_audio = h_va * x_audio
#     #     x_visual = h_av * x_visual
#     #     # print("音频 注意机制 x_audio = ", x_audio.shape)
#     #     # print("图像 注意机制 x_visual = ", x_visual.shape)
#     #
#     #     # 然后经过后面的2层1D卷积
#     #     x_audio = self.audio_model.forward_stage2(x_audio)
#     #     x_visual = self.visual_model.forward_stage2(x_visual)
#     #     # print("音频 stage2 x_audio = ", x_audio.shape)
#     #     # print("图像 stage2 x_visual = ", x_visual.shape)
#     #
#     #     # 使用后期Transformer
#     #     proj_x_a = x_audio.permute(0, 2, 1)
#     #     proj_x_v = x_visual.permute(0, 2, 1)
#     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
#     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
#     #     h_av = self.av(proj_x_v, proj_x_a) # Transformer
#     #     h_va = self.va(proj_x_a, proj_x_v)
#     #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
#     #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
#     #
#     #     # 池化层
#     #     audio_pooled = h_av.mean([1])  # mean accross temporal dimension
#     #     video_pooled = h_va.mean([1])
#     #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
#     #     # print("视频 池化 video_pooled = ", video_pooled.shape)
#     #
#     #     # 拼接特征
#     #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
#     #     x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
#     #
#     #     # 进行分类
#     #     x1 = self.classifier_1(x)
#     #     return x1
#
#     # # 辅助网络有包含中期Transformer块 添加co-attention 模型五
#     # def forward_own(self, x_audio, x_visual):
#     #     x_audio_aux = x_audio  # 音频辅助网络
#     #
#     #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
#     #     # print("音频 stage1 = ", x_audio.shape)
#     #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
#     #
#     #     x_visual_aux = x_visual  # 图像辅助网络
#     #
#     #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
#     #     # print("视频 stage1 = ", x_visual.shape)
#     #
#     #     # 辅助网络1阶段
#     #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
#     #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
#     #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
#     #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
#     #
#     #     # # # 辅助网络中期Attention
#     #     # proj_x_aa = x_audio_aux.permute(0, 2, 1)
#     #     # proj_x_vv = x_visual_aux.permute(0, 2, 1)
#     #     # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
#     #     # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
#     #     # if h_aa.size(1) > 1:
#     #     #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
#     #     # h_aa = h_aa.sum([-2])
#     #     # if h_vv.size(1) > 1:
#     #     #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
#     #     # h_vv = h_vv.sum([-2])
#     #     # x_audio_aux = h_aa * x_audio_aux
#     #     # x_visual_aux = h_vv * x_visual_aux
#     #
#     #     # 辅助网络2阶段
#     #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
#     #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
#     #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
#     #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
#     #
#     #     # 辅助网络转换
#     #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
#     #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
#     #     # print("转换形状之后\n")
#     #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
#     #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
#     #
#     #     # 辅助网络后期自注意
#     #     x_audio_aux = self.av(x_audio_aux, x_audio_aux)  # Transformer
#     #     x_visual_aux = self.va(x_visual_aux, x_visual_aux)
#     #     # print("经过Transformer后\n")
#     #     # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
#     #     # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
#     #
#     #     # 辅助网络转换
#     #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
#     #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
#     #     # print("转换形状之后\n")
#     #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
#     #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
#     #
#     #     # 辅助网池化层
#     #     audio_aux_pooled = x_audio_aux.mean([-1])  # mean accross temporal dimension
#     #     visual_aux_pooled = x_visual_aux.mean([-1])
#     #     # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
#     #     # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
#     #
#     #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
#     #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
#     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
#     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
#     #
#     #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
#     #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
#     #
#     #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
#     #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
#     #
#     #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
#     #     #                                                     [3 4 4 ]
#     #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
#     #
#     #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
#     #     if h_va.size(1) > 1:  # if more than 1 head, take average
#     #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
#     #
#     #     h_va = h_va.sum([-2])
#     #
#     #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
#     #     x_audio = h_va * x_audio
#     #     x_visual = h_av * x_visual
#     #     # print("音频 注意机制 x_audio = ", x_audio.shape)
#     #     # print("图像 注意机制 x_visual = ", x_visual.shape)
#     #
#     #     # 然后经过后面的2层1D卷积
#     #     x_audio = self.audio_model.forward_stage2(x_audio)
#     #     x_visual = self.visual_model.forward_stage2(x_visual)
#     #     # print("音频 stage2 x_audio = ", x_audio.shape)
#     #     # print("图像 stage2 x_visual = ", x_visual.shape)
#     #
#     #     # 使用后期Transformer
#     #     proj_x_a = x_audio.permute(0, 2, 1)
#     #     proj_x_v = x_visual.permute(0, 2, 1)
#     #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
#     #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
#     #     h_av = self.av(proj_x_v, proj_x_a)  # Transformer
#     #     h_va = self.va(proj_x_a, proj_x_v)
#     #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
#     #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
#     #
#     #     # 主干网络使用co-attention
#     #     h_co = self.co_av(h_av, h_va)
#     #     # 池化层
#     #     h_co_pool = h_co.mean([1])
#     #     # audio_pooled = h_av.mean([1])  # mean accross temporal dimension
#     #     # video_pooled = h_va.mean([1])
#     #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
#     #     # print("视频 池化 video_pooled = ", video_pooled.shape)
#     #
#     #     # 拼接特征
#     #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
#     #     # x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
#     #     x = torch.cat((audio_aux_pooled, h_co_pool, visual_aux_pooled), dim=-1)
#     #
#     #     # 进行分类
#     #     x1 = self.classifier_1(x)
#     #     return x1
#
#     def forward_own(self, x_audio, x_visual):
#         x_audio_aux = x_audio  # 音频辅助网络
#
#         x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
#         # print("音频 stage1 = ", x_audio.shape)
#         x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
#
#         x_visual_aux = x_visual  # 图像辅助网络
#
#         x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
#         # print("视频 stage1 = ", x_visual.shape)
#
#         # 辅助网络1阶段
#         x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
#         # print("辅助音频 stage1 = ", x_audio_aux.shape)
#         x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
#         # print("辅助视频 stage1 = ", x_visual_aux.shape)
#
#         # # # 辅助网络中期Attention
#         # proj_x_aa = x_audio_aux.permute(0, 2, 1)
#         # proj_x_vv = x_visual_aux.permute(0, 2, 1)
#         # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
#         # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
#         # if h_aa.size(1) > 1:
#         #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
#         # h_aa = h_aa.sum([-2])
#         # if h_vv.size(1) > 1:
#         #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
#         # h_vv = h_vv.sum([-2])
#         # x_audio_aux = h_aa * x_audio_aux
#         # x_visual_aux = h_vv * x_visual_aux
#
#         # 辅助网络2阶段
#         x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
#         # print("辅助音频 stage2 = ", x_audio_aux.shape)
#         x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
#         # print("辅助视频 stage2 = ", x_visual_aux.shape)
#
#         # 辅助网络转换
#         x_audio_aux = x_audio_aux.permute(0, 2, 1)
#         x_visual_aux = x_visual_aux.permute(0, 2, 1)
#         # print("转换形状之后\n")
#         # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
#         # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
#
#         # 辅助网络后期Transformer自注意
#         x_audio_aux = self.av(x_audio_aux, x_audio_aux)  # Transformer
#         x_visual_aux = self.va(x_visual_aux, x_visual_aux)
#         # print("经过Transformer后\n")
#         # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
#         # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
#
#         # 辅助网络转换
#         # x_audio_aux = x_audio_aux.permute(0, 2, 1)
#         # x_visual_aux = x_visual_aux.permute(0, 2, 1)
#         # print("转换形状之后\n")
#         # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
#         # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
#
#         # # 辅助网池化层
#         audio_aux_pooled = x_audio_aux.mean([1])  # mean accross temporal dimension
#         visual_aux_pooled = x_visual_aux.mean([1])
#         # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
#         # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
#
#         proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
#         proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
#         # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
#         # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
#
#         _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
#         _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
#
#         if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
#             h_av = torch.mean(h_av, axis=1).unsqueeze(1)
#
#         h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
#         #                                                     [3 4 4 ]
#         # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
#
#         # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
#         if h_va.size(1) > 1:  # if more than 1 head, take average
#             h_va = torch.mean(h_va, axis=1).unsqueeze(1)
#
#         h_va = h_va.sum([-2])
#
#         # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
#         x_audio = h_va * x_audio
#         x_visual = h_av * x_visual
#         # print("音频 注意机制 x_audio = ", x_audio.shape)
#         # print("图像 注意机制 x_visual = ", x_visual.shape)
#
#         # 然后经过后面的2层1D卷积
#         x_audio = self.audio_model.forward_stage2(x_audio)
#         x_visual = self.visual_model.forward_stage2(x_visual)
#         # print("音频 stage2 x_audio = ", x_audio.shape)
#         # print("图像 stage2 x_visual = ", x_visual.shape)
#
#         # 使用后期Transformer
#         proj_x_a = x_audio.permute(0, 2, 1)
#         proj_x_v = x_visual.permute(0, 2, 1)
#         # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
#         # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
#         h_av = self.av(proj_x_v, proj_x_a)  # Transformer
#         h_va = self.va(proj_x_a, proj_x_v)
#         # print("音频 Transformer之后 x_audio = ", h_av.shape)
#         # print("图像 Transformer之后 x_visual = ", h_va.shape)
#
#         # 主干网络使用co-attention
#         h_co = self.co_av(h_va, h_av)
#         # 池化层
#         h_co_pool = h_co.mean([1])
#         # audio_pooled = h_av.mean([1])  # mean accross temporal dimension
#         # video_pooled = h_va.mean([1])
#         # print("音频 池化 audio_pooled = ", audio_pooled.shape)
#         # print("视频 池化 video_pooled = ", video_pooled.shape)
#
#         # print(audio_aux_pooled.shape) # ([24, 128])
#         # print(visual_aux_pooled.shape) # ([24, 128])
#         # print(h_co_pool.shape) # ([24, 128])
#
#         # print(h_co.shape)
#         # print(x_visual_aux.shape)
#         # 先将主干网络的h_co_pool与图像分支进行Co-Attention
#         # x = self.co_vv(h_co, x_visual_aux)
#
#         # print(x.shape)
#         # 再将主干x与音频分支进行Co-Attention
#         # x = self.co_aa(x, x_audio_aux)
#         # print(x.shape)
#         # 拼接特征
#         # x = torch.cat((audio_pooled, video_pooled), dim=-1)
#         # x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
#         x = torch.cat((audio_aux_pooled, h_co_pool, visual_aux_pooled), dim=-1)
#         # x = x.mean([1])
#         # print(x.shape)
#
#         # 进行分类
#         x1 = self.classifier_1(x)
#         return x1
#
#
#
#
#
#
#
#
#
#

# -*- coding: utf-8 -*-
"""
Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
"""

import torch
import torch.nn as nn
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention
from models.co_attention import CoAttention


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True))


class EfficientFaceTemporal(nn.Module):  # EfficientFaceTemporal模型 这个模型是21年发表在AAI上的一个轻量级面部表情识别模型

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        # 这段代码定义了模型的一个卷积层，它包含一个2D卷积操作，一个批量化诡异操作，以及一个ReLu激活函数
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True), )

        # 这段代码更新通道数量，便于下一层处理
        input_channels = output_channels

        # 定义最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义模型三个结点，每个阶段包含一系列的InvertedResidual块
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # 局部特征提取器与调制器
        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        # 这几行代码定义了模型的第五个卷积层，包含一个2D卷积操作，一个批量归一化操作，以及一个ReLU激活函数
        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True), )

        # 四行代码定义了四个1D卷积快
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        # 图像辅助网络
        # 四行代码定义了四个1D卷积快
        self.conv1d_0_viusal_aux = conv1d_block(output_channels, 64)
        self.conv1d_1_visual_aux = conv1d_block(64, 64)
        self.conv1d_2_visual_aux = conv1d_block(64, 128)
        self.conv1d_3_visual_aux = conv1d_block(128, 128)

        # 线性分类器
        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        # 将每个样本图像保存为类的一个属性
        self.im_per_sample = im_per_sample

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global average pooling
        return x

    # 对输入的视频x应用一些列的卷积，池化，调制器，以及局部特征提取器，然后对结果进行平均池化，并返回池化后结果 对应于图中EfficientFace提取图像特征

    def forward_stage1(self, x):
        # Getting samples per batch
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0, 2, 1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    # 检查x输入的形状是否满足预期，然后对x进行重塑与置换，再对x应用两个1D卷积快，并返回结果 对应图中图像分支前2个1D卷积

    # 图像辅助网络
    def forward_stage1_visual_aux(self, x):
        # Getting samples per batch
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0, 2, 1)
        x = self.conv1d_0_viusal_aux(x)
        x = self.conv1d_1_visual_aux(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    # 对输入x应用两个1D卷积块并返回结果 对应图中图像分支后2个1D卷积

    # 图像辅助网络
    def forward_stage2_visual_aux(self, x):
        x = self.conv1d_2_visual_aux(x)
        x = self.conv1d_3_visual_aux(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    # 对输入x进行平均池化，减少维度，然后将池化后x输入分类器，并返回分类器输出

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x
    # 对输入的x先进性forward_features进行特征提取，然后依次执行forward_stage1，forward_stage2和forward_classifier
    # 这个就类比图中整个图像分支 （注意哈，相当于单模态的图像情感预测，没有注意机制，就是先调用forward_features提取图像特征，然后将特征通过
    # forward_stage1和forward_stage2，对应图中4个1D卷积，最后通过forward_classifier线性层输出预测情感）


# 接收两个参数，参数一模型，参数二路径 这里的路径是训练好模型的路径 实现的功能就是把训练好的模型参数加载进模型中
def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # 从path路径加载检查点
    pre_trained_dict = checkpoint['state_dict']  # 从检查点中获取状态
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')  # 打印提示信息，表示初始化EfficientNet模型
    model.load_state_dict(pre_trained_dict, strict=False)  # 将预训练模型加载值model中


# 参数一，类别数，参数二，人物，参数三，序列长度 创建一个EfficientFaceTemporal模型，并返回
def get_model(num_classes, task, seq_length):
    model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
    return model


# 参数一，输入通道数，参数二，输出通道数，参数三，卷积核大小，参数四，补偿，参数五，填充方法
# 创建一个1D卷积块，包含一1D卷积层，一批量归一化层，一ReLU激活函数，一最大池化层
def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


class AudioCNNPool(nn.Module):  # AudioCNNPool模型实例，用于处理音频数据

    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()

        # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
        input_channels = 10
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        self.lstm_1 = nn.LSTM(128, 64, dropout=0.4, batch_first=True, bidirectional=True)

        # 音频辅助网络
        # 定义四个1D卷积快，每个卷积块包含一1D卷积层，一批量归一化层，一ReLU激活函数，和一最大池化层
        input_channels = 10
        self.conv1d_0_audio_aux = conv1d_block_audio(input_channels, 64)
        self.conv1d_1_audio_aux = conv1d_block_audio(64, 128)
        self.conv1d_2_audio_aux = conv1d_block_audio(128, 256)
        self.conv1d_3_audio_aux = conv1d_block_audio(256, 128)

        # 定义一个线性分类层
        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    # 四个卷积，然后分类器，输出音频情感预测
    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    # 音频辅助网络
    def forward_stage1_audio_aux(self, x):
        x = self.conv1d_0_audio_aux(x)
        x = self.conv1d_1_audio_aux(x)
        return x

    def forward_stage2(self, x):
        x = x.transpose(2, 1)
        x, _ = self.lstm_1(x)
        x = x.transpose(2, 1)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    # 音频辅助网络
    def forward_stage2_audio_aux(self, x):
        x = x.transpose(2, 1)
        x, _ = self.lstm_1(x)
        x = x.transpose(2, 1)
        x = self.conv1d_2_audio_aux(x)
        x = self.conv1d_3_audio_aux(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1


class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None',
                 num_heads=1):  # 这里的pretr_ef就是预训练EfficientFace_Trained_on_AffectNet7.pth.tar
        super(MultiModalCNN, self).__init__()
        assert fusion in ['ia', 'it', 'lt'], print(
            'Unsupported fusion method: {}'.format(fusion))  # 这行代码检查fusion参数是否在['ia', 'it', 'lt']中。如果不在，就会打印一条错误消息。

        self.audio_model = AudioCNNPool(num_classes=num_classes)  # 这行代码创建了一个AudioCNNPool模型实例，用于处理音频数据。
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes,
                                                  seq_length)  # 这行代码创建了一个EfficientFaceTemporal模型实例，用于处理视觉数据

        init_feature_extractor(self.visual_model, pretr_ef)  # 这行代码使用预训练的模型初始化EfficientFaceTemporal模型

        e_dim = 128  # 嵌入维度
        input_dim_video = 128  # 视频输入维度
        input_dim_audio = 128  # 音频的输入维度
        self.fusion = fusion  # 融合方法

        # 通过不同的融合方法，创建对应注意力块
        # Transformer对应为AttentionBlock，而Attention机制对应为Attention()
        # 这里的lt就表示后期融合Transformer it表示中期融合Transformer
        # if fusion in ['lt', 'it']:
        #     if fusion  == 'lt':
        #         self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
        #         self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
        #     elif fusion == 'it':
        #         input_dim_video = input_dim_video // 2
        #         self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
        #         self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
        # # 这里ia表示中期融合的注意机制
        # elif fusion in ['ia']:
        #     input_dim_video = input_dim_video // 2
        #
        #     self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
        #     self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)

        # 后期Transformer
        self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
                                 num_heads=num_heads)
        self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
                                 num_heads=num_heads)

        # 辅助网络后期Transfromer
        self.aa1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_audio, out_dim=e_dim,
                                  num_heads=num_heads)
        self.vv1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_video, out_dim=e_dim,
                                  num_heads=num_heads)

        # 中期Attention
        input_dim_video = input_dim_video // 2
        self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                             num_heads=num_heads)
        self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
                             num_heads=num_heads)

        # # 辅助网络Attention
        self.aa = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                            num_heads=num_heads)
        self.vv = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_video, out_dim=input_dim_video,
                            num_heads=num_heads)

        # 主干网络的co-attention机制
        self.co_av = CoAttention(input_dim=128, hidden_dim=64)

        # 主干辅助aa的co-attention机制
        self.co_aa = CoAttention(input_dim=128, hidden_dim=64)

        # 主干辅助vv的co-attention机制
        self.co_vv = CoAttention(input_dim=128, hidden_dim=64)

        # self.classifier_1 = nn.Sequential( # 创建一个线性分类器
        #             nn.Linear(e_dim*2, num_classes), # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
        #         )

        self.classifier_1 = nn.Sequential(  # 创建一个线性分类器
            nn.Linear(e_dim * 3, num_classes),  # 注意哈，这里要求输入classifer1中的张量最后一个维度，也就是特征维度，必须是e_dim * 2
        )

    def forward(self, x_audio, x_visual):  # 是MultiModalCNN类的主要方法，它通过融合方法fusion来决定使用哪一个前线传播

        return self.forward_own(x_audio, x_visual)
        # if self.fusion == 'lt':
        #     return self.forward_transformer(x_audio, x_visual)
        #
        # elif self.fusion == 'ia':
        #     return self.forward_feature_2(x_audio, x_visual)
        #
        # elif self.fusion == 'it':
        #     return self.forward_feature_3(x_audio, x_visual)

    # PS：输入的音频特征与图像特征都是128维度

    # 这个是中期Transformer的前向传播
    def forward_feature_3(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频数据输入2个1D卷积中，得到128维度输出
        x_visual = self.visual_model.forward_features(x_visual)  # 将图像数据输入特征提取模块
        x_visual = self.visual_model.forward_stage1(x_visual)  # 在经过2个1D卷积，得到64维输出

        # 假设有一个3维张量，形状是(2, 3, 4)，其中的2表示由2个矩阵，每个矩阵有3行4列，如果使用permute(0, 2, 1)后，那么张量变为(2, 4, 3)
        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        # 这两行使用Transformer机制
        h_av = self.av1(proj_x_v, proj_x_a)  # 以音频为k v，视频为q，属于视频分支
        h_va = self.va1(proj_x_a, proj_x_v)  # 以视频为k v，音频为q，属于音频分支

        # 对于Transformer的输出，恢复其形状
        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        # 将Transformer的输出添加到分支上
        x_audio = h_av + x_audio
        x_visual = h_va + x_visual

        # 将音频与图像特征，通过后面2层1D卷积
        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        # 进行池化操作
        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = x_visual.mean([-1])

        # 将池化后的音频特征与图像特征拼接
        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        # 将拼接特征送入分类器中，进行情绪预测
        x1 = self.classifier_1(x)
        return x1

    # 这个是中期注意机制的前向传播
    # def forward_feature_2(self, x_audio, x_visual):
    #     x_audio = self.audio_model.forward_stage1(x_audio) # 将音频特征输入2层1D卷积中，得到128维输出
    #     x_visual = self.visual_model.forward_features(x_visual) # 将图像通过EfficientFace进行特征提取
    #     x_visual = self.visual_model.forward_stage1(x_visual) # 将提取图像特征送入2层1D卷积中得到64维输出
    #     print("stage1 audio = ", x_audio.shape)             # ([40, 128, 150])
    #     print("stage1 visual =", x_visual.shape)            # ([40, 64, 15])
    #     proj_x_a = x_audio.permute(0,2,1) # 调整矩阵行与列
    #     proj_x_v = x_visual.permute(0,2,1) # 调整矩阵行与列
    #     print("changeshape audio = ", proj_x_a.shape)       # ([40, 150, 128])
    #     print("changeshape visual =", proj_x_v.shape)       # ([40, 15, 64])
    #     _, h_av = self.av1(proj_x_v, proj_x_a) # 使用注意机制，属于音频分支
    #     _, h_va = self.va1(proj_x_a, proj_x_v) # 使用注意机制，属于视频分支
    #
    #     if h_av.size(1) > 1: #if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
    #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
    #
    #     h_av = h_av.sum([-2]) # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
    #                           #                                                     [3 4 4 ]
    #                           # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
    #
    #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
    #     if h_va.size(1) > 1: #if more than 1 head, take average
    #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
    #
    #     h_va = h_va.sum([-2])
    #
    #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
    #     x_audio = h_va*x_audio
    #     x_visual = h_av*x_visual
    #     print("afterattention audio = ", x_audio.shape)     # ([40, 128, 150])
    #     print("afterattention visual =", x_visual.shape)    # ([40, 64, 15])
    #     # 然后经过后面的2层1D卷积
    #     x_audio = self.audio_model.forward_stage2(x_audio)
    #     x_visual = self.visual_model.forward_stage2(x_visual)
    #     print("stage2 audio = ", x_audio.shape)             # ([40, 128, 144])
    #     print("stage2 visual =", x_visual.shape)            # ([40, 128, 15])
    #     # 池化层
    #     audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
    #     video_pooled = x_visual.mean([-1])
    #     print("pooled audio = ", audio_pooled.shape)        # ([40, 128])
    #     print("pooled visual =", video_pooled.shape)        # ([40, 128])
    #
    #     # 拼接特征
    #     x = torch.cat((audio_pooled, video_pooled), dim=-1)
    #
    #     # 进行分类
    #     x1 = self.classifier_1(x)
    #     return x1

    def forward_feature_2(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        _, h_av = self.av1(proj_x_v, proj_x_a)
        _, h_va = self.va1(proj_x_a, proj_x_v)

        if h_av.size(1) > 1:  # if more than 1 head, take average
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)

        h_av = h_av.sum([-2])

        if h_va.size(1) > 1:  # if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)

        h_va = h_va.sum([-2])

        x_audio = h_va * x_audio
        x_visual = h_av * x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = x_visual.mean([-1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        x1 = self.classifier_1(x)
        return x1

    # 这个是后期Transformer的前向传播
    def forward_transformer(self, x_audio, x_visual):
        print("进入后期Transformer\n")
        x_audio = self.audio_model.forward_stage1(x_audio)
        print("x_audio.shape stage1 = ", x_audio.shape)  # ([40, 128, 150]) 音频1
        proj_x_a = self.audio_model.forward_stage2(x_audio)  # 得到128维输出
        print("x_audio.shape stage2 = ", proj_x_a.shape)  # ([40, 128, 144]) 音频2

        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)
        print("x_visual.shape stage1 = ", x_visual.shape)  # ([40, 64, 15]) 视频1
        proj_x_v = self.visual_model.forward_stage2(x_visual)  # 得到128维输出
        print("x_visual.shape stage2 = ", proj_x_v.shape)  # ([40, 128, 15]) 视频2

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        print("转换形状之后\n")
        print("x_audio.shape shape = ", proj_x_a.shape)  # ([40, 144, 128]) 音频
        print("x_visual.shape shape = ", proj_x_v.shape)  # ([40, 15, 128]) 视频
        h_av = self.av(proj_x_v, proj_x_a)  # Transformer
        h_va = self.va(proj_x_a, proj_x_v)
        print("经过Transformer后\n")
        print("h_av.shape = ", h_av.shape)  # ([40, 144, 128]) 音频
        print("h_va.shape = ", h_av.shape)  # ([40, 144, 128]) 视频
        # 池化层
        audio_pooled = h_av.mean([1])  # mean accross temporal dimension
        video_pooled = h_va.mean([1])
        print("池化后\n")
        print("audio_pooled.shape = ", audio_pooled.shape)  # ([40, 128]) 音频
        print("video_pooled.shape = ", video_pooled.shape)  # ([40, 128]) 视频
        # 拼接特征
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        print("拼接后\n")
        print("x.shape = ", x.shape)  # ([40, 256])

        # 情感预测
        x1 = self.classifier_1(x)
        print("预测后\n")
        print("x1.shape= ", x1.shape)  # ([40, 8])
        return x1

    # 自己改的，想要Attention中期 单头注意机制 + Transformer后期 + 两个单模态，最后改下线性层的输入维度为e_dim * 4就好 然后把4个特征拼接起来
    # 但是希望在两个单模态的分支上，能够添加损失函数与中间的分支做交互，因为目的是为了补偿
    # 下面这个是没有辅助网络的，主干网络只有Attention和Transformer
    # def forward_own(self, x_audio, x_visual):
    #     x_audio_aux = x_audio # 音频辅助网络
    #
    #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
    #     # print("音频 stage1 = ", x_audio.shape)
    #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
    #
    #     x_visual_aux = x_visual # 图像辅助网络
    #
    #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
    #     # print("视频 stage1 = ", x_visual.shape)
    #
    #     # 辅助网络
    #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
    #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
    #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
    #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
    #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
    #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
    #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
    #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
    #
    #     # 辅助网池化层
    #     audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
    #     visual_aux_pooled = x_visual_aux.mean([-1])
    #     # # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)
    #     # # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)
    #
    #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
    #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
    #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
    #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
    #
    #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
    #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
    #
    #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
    #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
    #
    #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
    #     #                                                     [3 4 4 ]
    #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
    #
    #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
    #     if h_va.size(1) > 1:  # if more than 1 head, take average
    #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
    #
    #     h_va = h_va.sum([-2])
    #
    #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
    #     x_audio = h_va * x_audio
    #     x_visual = h_av * x_visual
    #     # print("音频 注意机制 x_audio = ", x_audio.shape)
    #     # print("图像 注意机制 x_visual = ", x_visual.shape)
    #
    #     # 然后经过后面的2层1D卷积
    #     x_audio = self.audio_model.forward_stage2(x_audio)
    #     x_visual = self.visual_model.forward_stage2(x_visual)
    #     # print("音频 stage2 x_audio = ", x_audio.shape)
    #     # print("图像 stage2 x_visual = ", x_visual.shape)
    #
    #     # 使用后期Transformer
    #     proj_x_a = x_audio.permute(0, 2, 1)
    #     proj_x_v = x_visual.permute(0, 2, 1)
    #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
    #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
    #     h_av = self.av(proj_x_v, proj_x_a) # Transformer
    #     h_va = self.va(proj_x_a, proj_x_v)
    #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
    #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
    #
    #     # 池化层
    #     audio_pooled = h_av.mean([1])  # mean accross temporal dimension
    #     video_pooled = h_va.mean([1])
    #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
    #     # print("视频 池化 video_pooled = ", video_pooled.shape)
    #
    #     # 拼接特征
    #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
    #     x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
    #
    #     # 进行分类
    #     x1 = self.classifier_1(x)
    #     return x1

    # # 辅助网络有包含中期Transformer块
    # def forward_own(self, x_audio, x_visual):
    #     x_audio_aux = x_audio # 音频辅助网络
    #
    #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
    #     # print("音频 stage1 = ", x_audio.shape)
    #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
    #
    #     x_visual_aux = x_visual # 图像辅助网络
    #
    #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
    #     # print("视频 stage1 = ", x_visual.shape)
    #
    #     # 辅助网络1阶段
    #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
    #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
    #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
    #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
    #
    #
    #     # # 辅助网络中期Attention
    #     # proj_x_aa = x_audio_aux.permute(0, 2, 1)
    #     # proj_x_vv = x_visual_aux.permute(0, 2, 1)
    #     # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
    #     # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
    #     # if h_aa.size(1) > 1:
    #     #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
    #     # h_aa = h_aa.sum([-2])
    #     # if h_vv.size(1) > 1:
    #     #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
    #     # h_vv = h_vv.sum([-2])
    #     # x_audio_aux = h_aa * x_audio_aux
    #     # x_visual_aux = h_vv * x_visual_aux
    #
    #     # 辅助网络2阶段
    #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
    #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
    #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
    #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
    #
    #     # 辅助网络转换
    #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
    #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
    #     # print("转换形状之后\n")
    #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
    #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
    #
    #     # 辅助网络后期自注意
    #     x_audio_aux = self.av(x_audio_aux, x_audio_aux) # Transformer
    #     x_visual_aux = self.va(x_visual_aux, x_visual_aux)
    #     # print("经过Transformer后\n")
    #     # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
    #     # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
    #
    #     # 辅助网络转换
    #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
    #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
    #     # print("转换形状之后\n")
    #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
    #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
    #
    #     # 辅助网池化层
    #     audio_aux_pooled = x_audio_aux.mean([-1]) #mean accross temporal dimension
    #     visual_aux_pooled = x_visual_aux.mean([-1])
    #     # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
    #     # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
    #
    #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
    #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
    #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
    #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
    #
    #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
    #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
    #
    #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
    #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
    #
    #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
    #     #                                                     [3 4 4 ]
    #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
    #
    #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
    #     if h_va.size(1) > 1:  # if more than 1 head, take average
    #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
    #
    #     h_va = h_va.sum([-2])
    #
    #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
    #     x_audio = h_va * x_audio
    #     x_visual = h_av * x_visual
    #     # print("音频 注意机制 x_audio = ", x_audio.shape)
    #     # print("图像 注意机制 x_visual = ", x_visual.shape)
    #
    #     # 然后经过后面的2层1D卷积
    #     x_audio = self.audio_model.forward_stage2(x_audio)
    #     x_visual = self.visual_model.forward_stage2(x_visual)
    #     # print("音频 stage2 x_audio = ", x_audio.shape)
    #     # print("图像 stage2 x_visual = ", x_visual.shape)
    #
    #     # 使用后期Transformer
    #     proj_x_a = x_audio.permute(0, 2, 1)
    #     proj_x_v = x_visual.permute(0, 2, 1)
    #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
    #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
    #     h_av = self.av(proj_x_v, proj_x_a) # Transformer
    #     h_va = self.va(proj_x_a, proj_x_v)
    #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
    #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
    #
    #     # 池化层
    #     audio_pooled = h_av.mean([1])  # mean accross temporal dimension
    #     video_pooled = h_va.mean([1])
    #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
    #     # print("视频 池化 video_pooled = ", video_pooled.shape)
    #
    #     # 拼接特征
    #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
    #     x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
    #
    #     # 进行分类
    #     x1 = self.classifier_1(x)
    #     return x1

    # # 辅助网络有包含中期Transformer块 添加co-attention 模型五
    # def forward_own(self, x_audio, x_visual):
    #     x_audio_aux = x_audio  # 音频辅助网络
    #
    #     x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
    #     # print("音频 stage1 = ", x_audio.shape)
    #     x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取
    #
    #     x_visual_aux = x_visual  # 图像辅助网络
    #
    #     x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
    #     # print("视频 stage1 = ", x_visual.shape)
    #
    #     # 辅助网络1阶段
    #     x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
    #     # print("辅助音频 stage1 = ", x_audio_aux.shape)
    #     x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
    #     # print("辅助视频 stage1 = ", x_visual_aux.shape)
    #
    #     # # # 辅助网络中期Attention
    #     # proj_x_aa = x_audio_aux.permute(0, 2, 1)
    #     # proj_x_vv = x_visual_aux.permute(0, 2, 1)
    #     # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
    #     # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
    #     # if h_aa.size(1) > 1:
    #     #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
    #     # h_aa = h_aa.sum([-2])
    #     # if h_vv.size(1) > 1:
    #     #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
    #     # h_vv = h_vv.sum([-2])
    #     # x_audio_aux = h_aa * x_audio_aux
    #     # x_visual_aux = h_vv * x_visual_aux
    #
    #     # 辅助网络2阶段
    #     x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
    #     # print("辅助音频 stage2 = ", x_audio_aux.shape)
    #     x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
    #     # print("辅助视频 stage2 = ", x_visual_aux.shape)
    #
    #     # 辅助网络转换
    #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
    #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
    #     # print("转换形状之后\n")
    #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
    #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
    #
    #     # 辅助网络后期自注意
    #     x_audio_aux = self.av(x_audio_aux, x_audio_aux)  # Transformer
    #     x_visual_aux = self.va(x_visual_aux, x_visual_aux)
    #     # print("经过Transformer后\n")
    #     # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
    #     # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频
    #
    #     # 辅助网络转换
    #     x_audio_aux = x_audio_aux.permute(0, 2, 1)
    #     x_visual_aux = x_visual_aux.permute(0, 2, 1)
    #     # print("转换形状之后\n")
    #     # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
    #     # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频
    #
    #     # 辅助网池化层
    #     audio_aux_pooled = x_audio_aux.mean([-1])  # mean accross temporal dimension
    #     visual_aux_pooled = x_visual_aux.mean([-1])
    #     # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
    #     # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])
    #
    #     proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
    #     proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
    #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
    #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])
    #
    #     _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
    #     _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支
    #
    #     if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
    #         h_av = torch.mean(h_av, axis=1).unsqueeze(1)
    #
    #     h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
    #     #                                                     [3 4 4 ]
    #     # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]
    #
    #     # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
    #     if h_va.size(1) > 1:  # if more than 1 head, take average
    #         h_va = torch.mean(h_va, axis=1).unsqueeze(1)
    #
    #     h_va = h_va.sum([-2])
    #
    #     # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
    #     x_audio = h_va * x_audio
    #     x_visual = h_av * x_visual
    #     # print("音频 注意机制 x_audio = ", x_audio.shape)
    #     # print("图像 注意机制 x_visual = ", x_visual.shape)
    #
    #     # 然后经过后面的2层1D卷积
    #     x_audio = self.audio_model.forward_stage2(x_audio)
    #     x_visual = self.visual_model.forward_stage2(x_visual)
    #     # print("音频 stage2 x_audio = ", x_audio.shape)
    #     # print("图像 stage2 x_visual = ", x_visual.shape)
    #
    #     # 使用后期Transformer
    #     proj_x_a = x_audio.permute(0, 2, 1)
    #     proj_x_v = x_visual.permute(0, 2, 1)
    #     # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
    #     # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
    #     h_av = self.av(proj_x_v, proj_x_a)  # Transformer
    #     h_va = self.va(proj_x_a, proj_x_v)
    #     # print("音频 Transformer之后 x_audio = ", h_av.shape)
    #     # print("图像 Transformer之后 x_visual = ", h_va.shape)
    #
    #     # 主干网络使用co-attention
    #     h_co = self.co_av(h_av, h_va)
    #     # 池化层
    #     h_co_pool = h_co.mean([1])
    #     # audio_pooled = h_av.mean([1])  # mean accross temporal dimension
    #     # video_pooled = h_va.mean([1])
    #     # print("音频 池化 audio_pooled = ", audio_pooled.shape)
    #     # print("视频 池化 video_pooled = ", video_pooled.shape)
    #
    #     # 拼接特征
    #     # x = torch.cat((audio_pooled, video_pooled), dim=-1)
    #     # x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
    #     x = torch.cat((audio_aux_pooled, h_co_pool, visual_aux_pooled), dim=-1)
    #
    #     # 进行分类
    #     x1 = self.classifier_1(x)
    #     return x1

    def forward_own(self, x_audio, x_visual):
        x_audio_aux = x_audio  # 音频辅助网络

        x_audio = self.audio_model.forward_stage1(x_audio)  # 将音频特征输入2层1D卷积中，得到128维输出
        # print("音频 stage1 = ", x_audio.shape)
        x_visual = self.visual_model.forward_features(x_visual)  # 将图像通过EfficientFace进行特征提取

        x_visual_aux = x_visual  # 图像辅助网络

        x_visual = self.visual_model.forward_stage1(x_visual)  # 将提取图像特征送入2层1D卷积中得到64维输出
        # print("视频 stage1 = ", x_visual.shape)

        # 辅助网络1阶段
        x_audio_aux = self.audio_model.forward_stage1_audio_aux(x_audio_aux)  # 将音频特征输入2层1D卷积中，得到128维输出
        # print("辅助音频 stage1 = ", x_audio_aux.shape)
        x_visual_aux = self.visual_model.forward_stage1_visual_aux(x_visual_aux)  # 将提取图像特征送入2层1D卷积中得到64维输出
        # print("辅助视频 stage1 = ", x_visual_aux.shape)

        # # # 辅助网络中期Attention
        # proj_x_aa = x_audio_aux.permute(0, 2, 1)
        # proj_x_vv = x_visual_aux.permute(0, 2, 1)
        # _, h_aa = self.aa(proj_x_aa, proj_x_aa)
        # _, h_vv = self.vv(proj_x_vv, proj_x_vv)
        # if h_aa.size(1) > 1:
        #     h_aa = torch.mean(h_aa, axis=1).unsqueeze(1)
        # h_aa = h_aa.sum([-2])
        # if h_vv.size(1) > 1:
        #     h_vv = torch.mean(h_vv, axis=1).unsqueeze(1)
        # h_vv = h_vv.sum([-2])
        # x_audio_aux = h_aa * x_audio_aux
        # x_visual_aux = h_vv * x_visual_aux

        # 辅助网络2阶段
        x_audio_aux = self.audio_model.forward_stage2_audio_aux(x_audio_aux)
        # print("辅助音频 stage2 = ", x_audio_aux.shape)
        x_visual_aux = self.visual_model.forward_stage2_visual_aux(x_visual_aux)
        # print("辅助视频 stage2 = ", x_visual_aux.shape)

        # 辅助网络转换
        x_audio_aux = x_audio_aux.permute(0, 2, 1)
        x_visual_aux = x_visual_aux.permute(0, 2, 1)
        # print("转换形状之后\n")
        # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
        # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频

        # 辅助网络后期Transformer自注意
        x_audio_aux = self.av(x_audio_aux, x_audio_aux)  # Transformer
        x_visual_aux = self.va(x_visual_aux, x_visual_aux)
        # print("经过Transformer后\n")
        # print("x_audio_aux.shape = ", x_audio_aux.shape)                              # ([40, 144, 128]) 音频
        # print("x_visual_aux.shape = ", x_visual_aux.shape)                              # ([40, 15, 128]) 视频

        # 辅助网络转换
        # x_audio_aux = x_audio_aux.permute(0, 2, 1)
        # x_visual_aux = x_visual_aux.permute(0, 2, 1)
        # print("转换形状之后\n")
        # print("x_audio_aux.shape shape = ", x_audio_aux.shape)                 # ([40, 144, 128]) 音频
        # print("x_visual_aux.shape shape = ", x_visual_aux.shape)                # ([40, 15, 128]) 视频

        # # 辅助网池化层
        audio_aux_pooled = x_audio_aux.mean([1])  # mean accross temporal dimension
        visual_aux_pooled = x_visual_aux.mean([1])
        # print("辅助音频 池化 audio_aux_pooled = ", audio_aux_pooled.shape)     # ([40, 128, 144])
        # print("辅助视频 池化 visual_aux_pooled = ", visual_aux_pooled.shape)   # ([40, 128, 15])

        proj_x_a = x_audio.permute(0, 2, 1)  # 调整矩阵行与列
        proj_x_v = x_visual.permute(0, 2, 1)  # 调整矩阵行与列
        # print("音频 转换 形状 x_audio = ", proj_x_a.shape)                  # ([40, 128])
        # print("视频 转换 形状 x_visual = ", proj_x_v.shape)                 # ([40, 128])

        _, h_av = self.av1(proj_x_v, proj_x_a)  # 使用注意机制，属于音频分支
        _, h_va = self.va1(proj_x_a, proj_x_v)  # 使用注意机制，属于视频分支

        if h_av.size(1) > 1:  # if more than 1 head, take average # 检查是否为多头注意力机制，如果是，就需要对h_va进行平均
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)

        h_av = h_av.sum([-2])  # 对张量倒数第二个维度进行求和，举个简单的例子，比如一个二维的矩阵[1 2 3 ]
        #                                                     [3 4 4 ]
        # 那么现在执行sum[-1]也就是对倒数第一个维度求和，由于矩阵是2 x 3，因此是按照列求和，得到结果是为 [4 6 7 ]

        # 同理的，本质上就是对多头注意机制，除以头的数量，得到的结果维度与单头输出结果形状一样
        if h_va.size(1) > 1:  # if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)

        h_va = h_va.sum([-2])

        # 由于注意机制输出的不是特征，而是一个权重，因此这里要进行相乘
        x_audio = h_va * x_audio
        x_visual = h_av * x_visual
        # print("音频 注意机制 x_audio = ", x_audio.shape)
        # print("图像 注意机制 x_visual = ", x_visual.shape)

        # 然后经过后面的2层1D卷积
        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)
        # print("音频 stage2 x_audio = ", x_audio.shape)
        # print("图像 stage2 x_visual = ", x_visual.shape)

        # 使用后期Transformer
        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)
        # print("音频 转换 形状 x_audio = ", proj_x_a.shape)
        # print("视频 转换 形状 x_visual = ", proj_x_v.shape)
        h_av = self.av(proj_x_v, proj_x_a)  # Transformer
        h_va = self.va(proj_x_a, proj_x_v)
        # print("音频 Transformer之后 x_audio = ", h_av.shape)
        # print("图像 Transformer之后 x_visual = ", h_va.shape)

        # 主干网络使用co-attention
        h_co = self.co_av(h_va, h_av)
        # 池化层
        h_co_pool = h_co.mean([1])
        # audio_pooled = h_av.mean([1])  # mean accross temporal dimension
        # video_pooled = h_va.mean([1])
        # print("音频 池化 audio_pooled = ", audio_pooled.shape)
        # print("视频 池化 video_pooled = ", video_pooled.shape)

        # print(audio_aux_pooled.shape) # ([24, 128])
        # print(visual_aux_pooled.shape) # ([24, 128])
        # print(h_co_pool.shape) # ([24, 128])

        # print(h_co.shape)
        # print(x_visual_aux.shape)
        # 先将主干网络的h_co_pool与图像分支进行Co-Attention
        # x = self.co_vv(h_co, x_visual_aux)

        # print(x.shape)
        # 再将主干x与音频分支进行Co-Attention
        # x = self.co_aa(x, x_audio_aux)
        # print(x.shape)
        # 拼接特征
        # x = torch.cat((audio_pooled, video_pooled), dim=-1)
        # x = torch.cat((audio_aux_pooled, audio_pooled, video_pooled, visual_aux_pooled), dim=-1)
        x = torch.cat((audio_aux_pooled, h_co_pool, visual_aux_pooled), dim=-1)
        # x = x.mean([1])
        # print(x.shape)

        # 进行分类
        x1 = self.classifier_1(x)
        return x1









