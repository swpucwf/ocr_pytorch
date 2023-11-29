import argparse
import yaml
from easydict import EasyDict as edict
from  alphabets import plateName,plate_chr



def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--img_h', type=int, default=48, help='height')
    parser.add_argument('--img_w' ,type=int ,default=168 ,help='width')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f ,Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = plateName
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.HEIGH T =args.img_h
    config.WIDTH = args.img_w
    return config
