# KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation

from os import path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append("..")

class DomainNetSet(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(DomainNetSet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


class DomainNetLoader:
    def __init__(
        self,
        domain_name='clipart',
        dataset_path=None,
        batch_size=64,
        num_workers=4,
        use_gpu=False,
        _C=None, 
    ):
        super(DomainNetLoader, self).__init__()
        self.domain_name = domain_name
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self._C = _C
        # -------domainbed----------
        # https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
        self.transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transforms_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def read_data(self, domain_name, split='train'):
        data_paths = []
        data_labels = []
        split_file = path.join(self.dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(self.dataset_path, data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
        return data_paths, data_labels

    def get_dloader(self):
        '''
        return the ##whole## training/val/test dataloader of the target domain
        '''
        print(f'==> Loading DomainNet {self.domain_name}...')

        # dataset_path = path.join(base_path, 'dataset', 'DomainNet')
        train_data_path, train_data_label = self.read_data(self.domain_name, split="train")
        val_data_path, val_data_label = self.read_data(self.domain_name, split="val")
        test_data_path, test_data_label = self.read_data(self.domain_name, split="test")

        train_dataset = DomainNetSet(train_data_path, train_data_label, self.transforms_train)
        val_dataset = DomainNetSet(val_data_path, val_data_label, self.transforms_test)
        test_dataset = DomainNetSet(test_data_path, test_data_label, self.transforms_test)

        train_dloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                                shuffle=True)
        val_dloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                                shuffle=False)
        test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                                shuffle=False)
        print(f'Train sample number: {len(train_data_path)}, Val sample number: {len(val_data_path)}, Test sample number: {len(test_data_path)}')
        return train_dloader, val_dloader, test_dloader


    def get_source_dloaders(self, domain_ls):
        '''
            load source domains
            return train/val list, which length = len(source_domains), each element is a source dataloader
        '''
        print(f"==> Loading dataset {domain_ls}")
        data_loader = {}
        iter_data_loader = {}
        for d in domain_ls:
            data_loader[d] = {}
            iter_data_loader[d] = {}

            train_data_path, train_data_label = self.read_data(d, split="train")
            train_dataset = DomainNetSet(train_data_path, train_data_label, self.transforms_train,)
            train_dloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
            data_loader[d]['train'] = train_dloader
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])

            val_data_path, val_data_label = self.read_data(d, split="val")
            val_dataset = DomainNetSet(val_data_path, val_data_label, self.transforms_test,)
            val_dloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
            data_loader[d]['val'] = val_dloader
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])

            test_data_path, test_data_label = self.read_data(d, split="test")
            test_dataset = DomainNetSet(test_data_path, test_data_label, self.transforms_test,)
            test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
            data_loader[d]['test'] = test_dloader
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])

            # print(len(train_dloader))
        return data_loader, iter_data_loader
    