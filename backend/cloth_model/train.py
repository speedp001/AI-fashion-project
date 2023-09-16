import argparse
import os
import random
import shutil
import time
import warnings
import pandas as pd
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
# add customdataset
from customdataset import MyDataset






model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet_b0',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./backend/cloth_model/weight/checkpoint.pt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')



parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
# 옵션은 여러 GPU가 있는 환경에서 특정 GPU를 선택하고자 할 때 사용
# ex) python script.py --gpu 0

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")




def main():
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        # cuda 연산 시에 무결성 보장 기능
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # print(args.distributed) -> single gpu면 False

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        # print(ngpus_per_node)
    else:
        ngpus_per_node = 1
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)




def main_worker(gpu, ngpus_per_node, args):
    
    # 모든 경고를 무시하려면
    warnings.filterwarnings("ignore")

    # # 경고를 다시 표시하려면
    # warnings.filterwarnings("default")
    
    global best_acc1
    best_acc1 = 0  # 초기화
    args.gpu = gpu

    # 그래프를 그리기 위해서 결과값을 저장하는 변수
    training_progress = {
        'train_losses': [],
        'train_accs': [],
        'valid_losses': [],
        'valid_accs': [],
    }

    ####### 특정 GPU 여부 확인
    if args.gpu is not None:
        # 특정 GPU 지정해서 사용 시 출력문
        print("Use GPU: {} for training".format(args.gpu))



    ####### Multi GPU 연산 확인
    if args.distributed:
        print("Multi GPU")
        
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else: print("Single GPU")
    
    
    
    ####### Create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        model.classifier[1] = nn.Linear(1280, out_features=21, bias=True)

        # # 모델 구조 출력 (옵션)
        # print(model)

    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False)
        model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        model.classifier[1] = nn.Linear(1280, out_features=21, bias=True)

        # # 모델 구조 출력 (옵션)
        # print(model)

    
    
    
    
    ####### 학습 GPU 설정
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            device = torch.device("cuda")
            model = model.to(device)

    
    
    
    ####### Device 설정
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
            print(f"{args.gpu} : CUDA")
        else:
            device = torch.device("cuda")
            print("GPU 가속기 : CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("GPU 가속기 : MPS")
    else:
        device = torch.device("cpu")
        print("CPU 연산")
        
        
        
    ####### Define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)





    ####### Optionally resume from a checkpoint
    if args.resume:
        
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            training_progress = checkpoint.get('training_progress', training_progress)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    
    




    ####### Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, ToTensorV2())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, ToTensorV2())
    
    else:        
        train_transform = A.Compose([
            A.Resize(width=480,height=640),
            A.HorizontalFlip(p=0.7),
            A.OneOf([
                    A.ShiftScaleRotate(shift_limit=0.0625),
                    A.ShiftScaleRotate(scale_limit=0.10), 
                    A.ShiftScaleRotate(rotate_limit=15),
            ], p=0.5),
            A.GaussianBlur(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        val_transform =A.Compose([
            A.Resize(width=480,height=640),
            A.OneOf([
                    A.ShiftScaleRotate(shift_limit=0.0625),
                    A.ShiftScaleRotate(scale_limit=0.10), 
                    A.ShiftScaleRotate(rotate_limit=15),
            ], p=0.7),
            A.GaussianBlur(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


        train_dataset = MyDataset("./backend/cloth_model/cloth_dataset/train/", transforms=train_transform)
        val_dataset = MyDataset("./backend/cloth_model/cloth_dataset/valid/", transforms=val_transform)

    # 분산 연산 시에 데이터 처리
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # 한 epoch에 대해 훈련
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device, args)
        training_progress['train_losses'].append(train_loss)
        training_progress['train_accs'].append(train_acc)

        # 검증 세트에서 평가
        valid_loss, valid_acc = validate(val_loader, device, model, criterion, args)
        training_progress['valid_losses'].append(valid_loss)
        training_progress['valid_accs'].append(valid_acc)


        scheduler.step()

        # remember best acc@1 and save checkpoint -> 이걸로 사용하시면 됩니다
        is_best = valid_acc > best_acc1
        best_acc1 = max(valid_acc, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'training_progress': training_progress,
            }, is_best)
        
            
        # # losses값과 accs값 test코드
        # print(train_losses, valid_losses)
        # print(train_accs, valid_accs)
        
    # 그래프를 그리는 함수 호출
    save_results_to_csv(training_progress['train_losses'], training_progress['valid_losses'], training_progress['train_accs'], training_progress['valid_accs'])
    plot_loss(training_progress['train_losses'], training_progress['valid_losses'])
    plot_accuracy(training_progress['train_accs'], training_progress['valid_accs'])


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top3],
        prefix="Epoch: [{}]".format(epoch+1))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # class 3 > class 2
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        #top1: 모델의 예측 중 가장 높은 확률 값을 가진 클래스와 실제 라벨이 일치하는 경우를 측정합니다. 이는 단일 예측의 정확도를 의미합니다.
        #top3: 모델의 예측 중 상위 3개 확률 값에 해당하는 클래스 중 실제 라벨이 있는지 여부를 측정합니다. 이는 상위 3개 예측 중 하나가 정답인 경우의 정확도를 의미합니다.
        #상위 3개의 얘측도 측정
        losses.update(loss.item(), images.size(0))
        
        acc1_float = acc1[0].item()
        acc3_float = acc3[0].item()
        top1.update(acc1_float, images.size(0))
        top3.update(acc3_float, images.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
            
    return losses.avg, top1.avg


def validate(val_loader, device, model, criterion, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to(device)
                    target = target.to(device)
                    # Move tensors to CPU before using them
                if torch.cuda.is_available():
                    images = images.to(device)
                    target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc3 = accuracy(output, target, topk=(1, 3))
                losses.update(loss.item(), images.size(0))

                acc1_float = acc1[0].item()
                acc3_float = acc3[0].item()
                top1.update(acc1_float, images.size(0))
                top3.update(acc3_float, images.size(0))


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top3 = AverageMeter('Acc@3', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top3],
        prefix='Valid: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top3.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return losses.avg, top1.avg  # Return top1 accuracy and loss



def save_checkpoint(state, is_best, filename='./backend/cloth_model/weight/checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './backend/cloth_model/weight/model_best.pt')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_results_to_csv(train_losses, train_accs, valid_losses, valid_accs):
    df = pd.DataFrame({
        'Train Loss': train_losses,
        'Train ACC': train_accs,
        'Validation Loss': valid_losses,
        'Validation ACC': valid_accs
    })
    df.to_csv('./backend/cloth_model/visualization/train_val_result.csv', index=False)

def plot_loss(train_losses, valid_losses):
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(valid_losses, label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./backend/cloth_model/visualization/loss_plot.jpg")

def plot_accuracy(train_accs, valid_accs):
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(valid_accs, label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("./backend/cloth_model/visualization/accuracy_plot.jpg")


if __name__ == '__main__':
    main()
