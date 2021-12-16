import torchvision.transforms as transforms
import argparse
import sys
import torch
import os
from torch.utils.data import DataLoader
from torch import is_distributed, optim,nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Normalize
sys.path.append('../model')
sys.path.append('../dataset')
from model import getmodel
from dataset import getdata_prefix_and_label,VideoDataset,ViewDataset
import trainutils,json

parser = argparse.ArgumentParser(description='train and validation paramaters')
parser.add_argument('-o','--out_dir',type=str,help='output dir',default='fir_exp')
parser.add_argument('-m','--mode',type=str,help="train mode or test mode") 
parser.add_argument('--model_type',type=str,help='model type',default='rnn')
parser.add_argument('--batch_size',type=int,help='batch size',default=8)
parser.add_argument('--load_weights',type=str,default=None)
parser.add_argument('--is_distributed', action='store_true', default=False, help='DistributedDataParallel or not')
parser.add_argument('--fine_tune',action='store_true',default=False)
parser.add_argument('--fix_feature_extractor',action='store_true',default=False)
parser.add_argument("--local_rank", default=-1,type=int)
args = parser.parse_args()

local_rank = args.local_rank

def clean_up():
    if args.is_distributed:
        dist.destroy_process_group()

def set_up(rank,world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def collate_fn_r3d_18(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor

def collate_fn_rnn(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor,labels_tensor

def main():
    print("enter main function")
    model = getmodel(model_type = args.model_type,num_classes = 17)
    # if args.load_weights is not None:
    #     model.load_state_dict(torch.load(args.load_weights,map_location=torch.device("cpu")))
    if args.is_distributed:
            # These are the parameters used to initialize the process group
        env_dict = {key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )
        n = 1
        device_ids = list(range(local_rank * n, (local_rank + 1) * n))
        print(f"device ids:{device_ids}")
        print(
            f"[{os.getpid()}] rank = {dist.get_rank()}, "
            + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )
        device = torch.device("cuda", local_rank)
        model = model.to(device)
        model = DDP(model, device_ids=device_ids)
        if args.load_weights is not None:
            print(f'rank={local_rank} start to load weights')
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
            model.module.load_state_dict(torch.load(args.load_weights,map_location=map_location))
            dist.barrier()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if args.load_weights is not None:
            print(f"Loading weights from {args.load_weights}")
            if not args.fine_tune:
                model.load_state_dict(torch.load(args.load_weights))
            else:
                pretrained_dict = torch.load(args.load_weights)
                model_dict = model.state_dict()
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k not in ["fc1.weight","fc1.bias"]}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
        if args.fine_tune and args.fix_feature_extractor:
            if args.load_weights is None:
                raise ValueError("load weights should not be None if at fine tune mode")
            for k,v in model.named_parameters():
                if k not in ["fc1.weight","fc1.bias"]:
                    v.requires_grad = False

    if args.model_type == 'rnn':
        h, w =224, 224
        mean = [15.7783325, 15.2805319, 14.89863584]
        std = [34.27505085839842, 33.18109585891682, 32.28101894712342]
    else:
        h, w = 112, 112
        # mean = [0.43216, 0.394666, 0.37645]
        # std = [0.22803, 0.22145, 0.216989]
    # train_transformer = transforms.Compose([
    #         transforms.Resize((h,w)),
    #         # transforms.RandomHorizontalFlip(p=0.5),    
    #         transforms.ToTensor(),
    #         ])
    train_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.horizontalFlip,
            transforms.ToTensor(),
            ])
    test_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            ])
    # trainprefix_list,trainload_name_txt,trainload_label_txt = getdata_prefix_and_label('train')
    # testprefix_list,testload_name_txt,testload_label_txt = getdata_prefix_and_label('test')
    # train_ds = VideoDataset(trainprefix_list,trainload_name_txt,trainload_label_txt,train_transformer)
    # test_ds = VideoDataset(testprefix_list,testload_name_txt,testload_label_txt,test_transformer)
    with open("../../viewcls_framedata/train_annotation.json",'r') as f:
        train_dic = json.load(f)
    with open("../../viewcls_framedata/test_annotation.json",'r') as f:
        test_dic = json.load(f)
    train_ds = ViewDataset(train_dic,transform=train_transformer)
    test_ds = ViewDataset(test_dic,transform=test_transformer)


    batch_size = args.batch_size
    # print(f"bacth size:{batch_size}")
    if args.model_type == "rnn":
        if args.is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
            train_dl = DataLoader(train_ds, batch_size= batch_size,
                        shuffle=False, collate_fn= collate_fn_rnn,sampler=train_sampler)
            test_dl = DataLoader(test_ds, batch_size= 2*batch_size,
                        shuffle=False, collate_fn= collate_fn_rnn,sampler=test_sampler)
        else:
            train_dl = DataLoader(train_ds, batch_size= batch_size,
                        shuffle=True, collate_fn= collate_fn_rnn)
            test_dl = DataLoader(test_ds, batch_size= 2*batch_size,
                        shuffle=False, collate_fn= collate_fn_rnn)  
    else:
        if args.is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
            train_dl = DataLoader(train_ds, batch_size= batch_size,
                        shuffle=False, collate_fn= collate_fn_r3d_18,sampler=train_sampler)
            test_dl = DataLoader(test_ds, batch_size= 2*batch_size,
                        shuffle=False, collate_fn= collate_fn_r3d_18,sampler=test_sampler)
        else:
            train_dl = DataLoader(train_ds, batch_size= batch_size,
                        shuffle=True, collate_fn= collate_fn_r3d_18)
            test_dl = DataLoader(test_ds, batch_size= 2*batch_size,
                        shuffle=False, collate_fn= collate_fn_r3d_18)
    
    loss_func = nn.CrossEntropyLoss(reduction="sum").to(device)
    if args.fine_tune and args.fix_feature_extractor:
        # opt = optim.Adam(params=[model.fc1.weight,model.fc1.bias],lr=3e-4,weight_decay=1e-5)
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=3e-4,weight_decay=1e-5)
    else:
        opt = optim.Adam(model.parameters(), lr=3e-4)
    lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=5,verbose=1)
    store_path = os.path.join('../../modeloutput',args.out_dir)
    os.makedirs(store_path, exist_ok=True)
    
    if args.is_distributed and dist.get_rank() == 0:
        # print("is distributed")
        writer = SummaryWriter(os.path.join(store_path,'log_dir'))
    else:
        writer = SummaryWriter(os.path.join(store_path,'log_dir'))

    # for k,v in model.named_parameters():
    #     if k not in ["fc1.weight","fc1.bias"]:
    #         print(v.requires_grad)

    params_train={
    "num_epochs": 200,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": test_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": os.path.join(store_path,args.model_type+".pt"),
    "device":device,
    "is_distributed":args.is_distributed,
    }
    print('start to train')
    trainutils.train_val(model,params_train,writer)
    # model,loss_hist,metric_hist = trainutils.train_val(model,params_train,writer)
    # trainutils.plot_loss(loss_hist, metric_hist, store_path)
    clean_up()

if __name__ == '__main__':
    main()