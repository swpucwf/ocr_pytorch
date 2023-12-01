import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

__all__ = ['CRNN', 'resnet_ocr']


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# myCfg = [32,'M',64,'M',128,'M',256]
# cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256]

class resnet_ocr(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False,in_channel=3):
        super(resnet_ocr, self).__init__()
        self.in_channel = in_channel
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
            # cfg =[32,32,'M',64,64,'M',128,128,'M',256,256]
        self.feature = self.make_layers(cfg, True)
        self.export = export
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(256, num_classes, 1, 1)

        self.ada_pool = nn.AdaptiveAvgPool2d((1,None))


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = self.in_channel
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else:
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.loc(x)
        x = self.newCnn(x)
        if self.export:  # 推理模式
            # x = self.ada_pool(x)
            conv = x.squeeze(2)  # b *512 * width
            # print("conv",conv.shape)
            conv = conv.transpose(2, 1)  # [w, b, c]
            conv = conv.argmax(dim=2)
            return conv
        else:  # 训练模式
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            # x = self.ada_pool(x)
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # output = F.log_softmax(self.rnn(conv), dim=2)
            output = F.log_softmax(conv, dim=2)
            return output

if __name__ == '__main__':
    #  model = CRNN(32, 3, 79,10)
    # 输入长度为[48, 168]
    # cfg = [32, 'M', 64, 'M', 128, 'M', 256]
    # 导出模式
    model = resnet_ocr(num_classes=78, export=True,in_channel=1)
    input =torch.FloatTensor(1, 1, 48, 168)
    #  torch.save(model.state_dict,"test.pth")
    out = model(input)
    print(out.size())
