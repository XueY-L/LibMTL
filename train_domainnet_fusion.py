import torch, argparse
import torch.nn as nn
import torch.nn.functional as F

from data.domainnet import DomainNetLoader
from utils.load_model import load_model

from trainer_fusion import Trainer_fusion
from LibMTL.model import resnet18
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric

LibMTL_args.add_argument('--dataset', default='domainnet', type=str, help='domainnet, office-home')
LibMTL_args.add_argument('--bs', default=32, type=int, help='batch size')
LibMTL_args.add_argument('--epochs', default=100, type=int, help='training epochs')
LibMTL_args.add_argument('--dataset_path', default='/home/yxue/datasets/DomainNet/', type=str, help='dataset path')

params = LibMTL_args.parse_args()
# set device
set_device(params.gpu_id)
# set random seed
set_random_seed(params.seed)

kwargs, optim_param, scheduler_param = prepare_args(params)

task_name = ['infograph', 'painting', 'quickdraw', 'real', 'sketch']

# define tasks
task_dict = {task: {'metrics': ['Acc'],
                    'metrics_fn': AccMetric(),
                    'loss_fn': CELoss(),
                    'weight': [1]}   # weight用在count_improvement方法中，代表metrics越高越好
                    for task in task_name}  

# prepare dataloaders
data_loader, _ = DomainNetLoader(
    dataset_path=params.dataset_path,
    batch_size=params.bs,
    num_workers=0,
).get_source_dloaders(domain_ls=[x for x in task_name if x != 'clipart'])

train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
test_dataloaders = {task: data_loader[task]['test'] for task in task_name}

# source models
param_ls = load_model(domain_ls=task_name)

# define encoder and decoders
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        hidden_dim = 512
        self.resnet_network = resnet18(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                    nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        
    def forward(self, inputs):
        out = self.resnet_network(inputs)
        out = torch.flatten(self.avgpool(out), 1)
        out = self.hidden_layer(out)
        return out

# encoder共享，decoder每个域有一个
decoders = nn.ModuleDict({task: nn.Sequential(nn.Linear(512, len(task_name)), nn.Softmax(dim=1)) for task in list(task_dict.keys())})

MTL_trainer = Trainer_fusion(task_dict=task_dict, 
                        weighting=weighting_method.__dict__[params.weighting], 
                        architecture=architecture_method.__dict__[params.arch], 
                        encoder_class=Encoder, 
                        decoders=decoders,
                        rep_grad=params.rep_grad,
                        multi_input=params.multi_input,
                        optim_param=optim_param,
                        scheduler_param=scheduler_param,
                        save_path=params.save_path,
                        load_path=params.load_path,
                        param_ls=param_ls,
                        **kwargs)
MTL_trainer.train(train_dataloaders=train_dataloaders, 
                      val_dataloaders=val_dataloaders,
                      test_dataloaders=test_dataloaders, 
                      epochs=params.epochs)