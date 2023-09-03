import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tmodels_resnet50_multidomain_paramls import resnet50
from utils.lerp import lerp_multi

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters

class Trainer_fusion(nn.Module):
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, 
                 save_path=None, load_path=None, param_ls=None, **kwargs):
        super(Trainer_fusion, self).__init__()
        
        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad  # computing gradient for representation or sharing parameters
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path
        self.param_ls = param_ls
        self.fusion_model = resnet50()

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)
        
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()  # pass?
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        
        # 输出模型参数量
        count_parameters(self.model)
        
    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad,
            'rmsprop': torch.optim.RMSprop,
        }
        scheduler_dict = {
            'exp': torch.optim.lr_scheduler.ExponentialLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
            'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader, reset_iter=False):
        if reset_iter:
            loader[1] = iter(loader[0])
        data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)  # 返回用preds和gts算的交叉熵损失
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        # train_loader是每个域的字典，其中values是dataloader和dataloader迭代器
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        print(f'每个域batch数量{train_batch}')
        train_batch = min(train_batch) if self.multi_input else train_batch  #为什么要取max？
        print(f'每个域取{train_batch}个batch')
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])  # 给每个batch一个值？这里放什么
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            reset_iter = True  # 每个epoch结束后，loader[1]迭代器会耗尽，此时需要用loader[0]重新制作迭代器
            for batch_index in range(train_batch):  # 第batch_index个batch
                # multi_input
                train_losses = torch.zeros(self.task_num).to(self.device)  # 存放每个task的loss
                for task_idx, task in enumerate(self.task_name):
                    train_input, train_gt = self._process_data(train_loader[task], reset_iter)  # torch.Size([bs, 3, 224, 224]) torch.Size([bs])
                    
                    # train_pred = self.model(train_input, task)  # {'task(domain_name)': torch.Size([bs, 345])}
                    # train_pred = train_pred[task]  # 从字典里提取预测结果
                    # train_losses[task_idx] = self._compute_loss(train_pred, train_gt, task)  # 对前两个算交叉熵
                    # self.meter.update(train_pred, train_gt, task)  # 更新meter.metrics，即accuracy，上一个是loss

                    # 聚合模型
                    weights = self.model(train_input, task)  # 字典，记录每个task的权重
                    www = torch.mean(weights[task], dim=0)  # 对batch内所有样本取平均
                    print(task, www)
                    target_param = lerp_multi(self.param_ls, www)

                    train_pred = self.fusion_model(train_input, target_param)
                    train_losses[task_idx] = self._compute_loss(train_pred, train_gt, task)
                    self.meter.update(train_pred, train_gt, task)

                reset_iter = False

                self.optimizer.zero_grad()
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])  # train_losses是每个任务的loss，拿去做反向传播
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w  # len=任务数的一维矩阵，不知道返回的是啥
                self.optimizer.step()
            
            self.meter.record_time('end')
            self.meter.get_score()  # 计算self.metrics, self.losses
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()  # 初始化，以记录下一个epoch
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            # self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:  # 在self.test的self.meter.display中，会更新每个val结果相对于base_result高多少，取高得最多的（取所有任务的improvement平均）
                # torch.save(self.model.state_dict(), os.path.join(self.save_path, f'best_epoch{epoch}.pt'))
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f'test_fusion.pt'))
                print('Save Model {} to {}\n'.format(epoch, os.path.join(self.save_path, 'test_fusion.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight

    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                test_loader[task][1] = iter(test_loader[task][0])
                for batch_index in range(test_batch[tn]):
                    test_input, test_gt = self._process_data(test_loader[task])
                    test_pred = self.model(test_input, task)
                    test_pred = test_pred[task]
                    test_loss = self._compute_loss(test_pred, test_gt, task)
                    self.meter.update(test_pred, test_gt, task)  # 更新meter.metrics，即accuracy，上一个是loss（不需要）
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement
