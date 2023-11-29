import argparse
import os

import torch
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader

import models
from  alphabets import plateName,plate_chr
from dataset import get_dataset
from utils import utils, function
from utils.utils import model_info


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    # 配置文件
    parser.add_argument('--cfg', help='experiment configuration filename', default="config/myocr.yaml", type=str)
    parser.add_argument('--img_h', type=int, default=48, help='height')
    parser.add_argument('--img_w' ,type=int ,default=168 ,help='width')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f ,Loader=yaml.FullLoader)
        config = edict(config)
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    # print(config.MODEL.NUM_CLASSES)
    # #  宽度高度
    config.HEIGHT =args.img_h
    config.WIDTH = args.img_w

    print(config)
    return config
def main():

    # load config
    config = parse_arg()
    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn 配置
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = models.get_model(config,cfg=eval(config.cfg))

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    loss_fn = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)

        for name, param in model.named_parameters():
            # 冻结权重层
            if name.startswith("feature"):
                param.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)
    model_info(model)

    train_dataset = get_dataset(config)(config, input_w=config.WIDTH,input_h=config.HEIGHT,is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config,input_w=config.WIDTH,input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model,
                       loss_fn, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()

        acc = function.validate(config, val_loader, val_dataset, converter,
                                model,loss_fn, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        # save checkpoint
        torch.save(
            {
                "cfg":eval(config.cfg),
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()