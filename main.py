# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import time


if __name__ == '__main__':
    opt = parse_opts() # 解析opts.py中的参数
    n_folds = 1 # 设置折叠数量，用于交叉验证
    # 如果n_folds = 5，那么数据集将被分成5个子集。然后我们进行5次学习试验，每次试验中，我们都从5个子集中选择一个作为验证集，其余4个子集组合成训练集。我们在训练集上训练模型，并在验证集上测试模型。我们重复这个过程5次，每次选择一个不同的验证集。
    # 在你的代码中，n_folds被设置为1。这意味着没有进行交叉验证，因为只有一个子集被用作训练集和验证集。这可能是因为你的数据集已经被预先分割成了训练集和验证集
    test_accuracies = [] # 创建一个空列表用于保存测试精度
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(opt.device)

    pretrained = opt.pretrain_path != 'None' # 这个就对应当前目录下EfficientFace_Trained_on_AffectNet7.pth.tar
    
    #opt.result_path = 'res_'+str(time.time())
    if not os.path.exists(opt.result_path): # 如果路径不存在
        os.makedirs(opt.result_path) # 创建结果路径
        
    opt.arch = '{}'.format(opt.model) # 设置架构为模型的格式化字符串
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)]) # 设置存储名臣为数据集、模型和采样持续时间的连续字符
            
    for fold in range(n_folds): # 对于每一折叠
        #if opt.dataset == 'RAVDESS':
        #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'

        print(opt) # 打印当前选项
        with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file) # 将选项变量写入文件
            
        torch.manual_seed(opt.manual_seed) # 设置pytorch随机种子
        model, parameters = generate_model(opt) # 生成模型与参数

        criterion = nn.CrossEntropyLoss() # 设置损失函数为交叉损失函数
        criterion = criterion.to(opt.device) # 将损失函数移动到设备上
        
        if not opt.no_train: # 如果需要训练
            
            video_transform = transforms.Compose([ # 创建一个视屏转换器
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])
        
            training_data = get_training_set(opt, spatial_transform=video_transform)  # 获取训练集 在dataset.py中，而在get_training_set()函数里，本质上调用的是ravdess.py
        
            train_loader = torch.utils.data.DataLoader( # 创建一个数据加载器
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            train_logger = Logger( # 创建一个训练日志记录器
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger( # 创建一个训练批次日志记录器
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
            

            optimizer = optim.SGD( # 创建一个SGC优化器
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
            
        if not opt.no_val: # 如果需要验证
            video_transform = transforms.Compose([ # 创建一个视频转换器
                transforms.ToTensor(opt.video_norm_value)])     
        
            validation_data = get_validation_set(opt, spatial_transform=video_transform) # 获取验证集
            
            val_loader = torch.utils.data.DataLoader( # 创建一个数据加载器
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            val_logger = Logger( # 创建一个验证日志记录器
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            test_logger = Logger( # 创建一个测试日志记录器
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            
        best_prec1 = 0 # 设置最好的精度 0
        best_loss = 1e10 # 设置最好的损失1e10
        if opt.resume_path: # 如果恢复路径
            print('loading checkpoint {}'.format(opt.resume_path)) # 打印加载检查点的信息
            checkpoint = torch.load(opt.resume_path) # 加载检查点
            assert opt.arch == checkpoint['arch'] #
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)

                # 注意哈，这里是训练模型的地方tran_epoch就是训练模型
                # 从这里进入tarin.py中tarin_epoch函数中

                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, False, opt, fold)
            
            if not opt.no_val:
                
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = { # 创建一个字典来保存模型的状态
                'epoch': i, # 当前的周期
                'arch': opt.arch, # 模型的架构
                'state_dict': model.state_dict(), # 模型的参数
                'optimizer': optimizer.state_dict(), # 模型的优化器参数
                'best_prec1': best_prec1 # 目前最佳精度
                }
               
                save_checkpoint(state, is_best, opt, fold) # 保存模型检查点

               
        if opt.test: # 如果需要测试

            test_logger = Logger( # 创建一个日志记录器
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            video_transform = transforms.Compose([ # 创建一个视频转换器
                transforms.ToTensor(opt.video_norm_value)])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) # 获取测试集
        
            #load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth') # 加载当前最佳模型
            model.load_state_dict(best_state['state_dict']) # 加载最好的模型参数
        
            test_loader = torch.utils.data.DataLoader( # 创建一个数据加载器
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt, # 进行以此测试周期，返回测试损失和精度
                                            test_logger)
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f: # 打开一个文件用于追加写入
                    f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss)) # 将精度和损失写入
            test_accuracies.append(test_prec1)  # 将精度添加到精度列表中
                
            
    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f: # 打卡文件用于追加写入
        f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n') # 将精度列表的平均值和标准差写入文件
