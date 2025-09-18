'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch)) # 打印当前训练周期

    # 将模型设置训练模式
    model.train()

    batch_time = AverageMeter() # 计算处理时间平均值
    data_time = AverageMeter() # 计算数据加载时间的平均值
    losses = AverageMeter() # 计算损失的平均值
    top1 = AverageMeter() # 计算top1精度
    top5 = AverageMeter() # 计算top5精度
        
    end_time = time.time() # 获取当前时间，从而计算批处理时间和数据加载时间
    # 开始一个循环，遍历数据加载器中的所有批次，每个批次中音频输入，视频输入和标签 target就是标签
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # 将数据移动到GPU上
        targets = targets.to(opt.device)

        # 如果参数不为空，注意哈，noise就是给数据集加噪声，
        if opt.mask is not None:
            with torch.no_grad():

                # 加噪声
                if opt.mask == 'noise':
                    # 拼接：音频输入 + 噪声 + 音频输入
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)
                    # 拼接：图像输入 + 图像输入 + 噪声
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0)
                    # 通过上面的拼接，就相当于把3个音频特征拼接在一起，因此这里是把3个标签拼接在啦一起
                    targets = torch.cat((targets, targets, targets), dim=0)
                    # 将拼接的音频随机排序
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    # 将图像按照音频的排序进行重排
                    visual_inputs = visual_inputs[shuffle]
                    # 将标签也按照音频的方式排序
                    targets = targets[shuffle]

                #
                elif opt.mask == 'softhard':
                    coefficients = torch.randint(low=0, high=100,size=(audio_inputs.size(0),1,1))/100 # 生成一个随机系数矩阵 （音频随机矩阵）
                    vision_coefficients = 1 - coefficients # 这里对系数矩阵进行减1操作，得到视觉随机举证
                    coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2)) # 将系数矩阵（音频随机矩阵）弄成与音频同样大小的形状
                    # 将视频随机矩阵与图像弄成相同形状
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))
                    # 将音频输入 + 音频输入 * 随即系数矩阵 + 零矩阵 + 输入音频矩阵 拼接
                    audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size()), audio_inputs), dim=0)
                    # 将图像 + 图像 * 视觉随机矩阵 + 图像 + 零矩阵 拼接
                    visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())), dim=0)
                    # 将4个目标进行拼接
                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    # 随机调整顺序
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
   


        visual_inputs = visual_inputs.permute(0,2,1,3,4) # 重新排列视觉输入维度
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        
        audio_inputs = Variable(audio_inputs) # 将音频输入转化为pytorch变量，以计算梯度
        visual_inputs = Variable(visual_inputs) # 将图像输入转化为pytorch变量，以计算梯度

        targets = Variable(targets) # 将标签转化为pytorch变量，以计算梯度
        outputs = model(audio_inputs, visual_inputs) # 使用模型进行预测，并得到预测结果
        loss = criterion(outputs, targets) # 计算结果与标签之间损失

        losses.update(loss.data, audio_inputs.size(0)) # 更新损失平均值
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5)) # 计算top1 top5之间精度
        top1.update(prec1, audio_inputs.size(0)) # 更新top1精度平均值
        top5.update(prec5, audio_inputs.size(0)) # 更新top5精度平均值

        optimizer.zero_grad() # 将优化器梯度去清零
        loss.backward() # 计算损失函数梯度
        optimizer.step() # 更新模型代码

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

 
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    
    if opt.model == 'multimodalcnn':
        # 调用train_epoch_multimodal函数进行模型训练
        train_epoch_multimodal(epoch,  data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger)
        return
    
    
