import argparse
import yaml
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from torch.backends import cudnn

from  alphabets import plateName,plate_chr
from utils import utils


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    # 配置文件
    parser.add_argument('--cfg', help='experiment configuration filename', default="config/crnn.yaml", type=str)
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


if __name__ == '__main__':
    main()