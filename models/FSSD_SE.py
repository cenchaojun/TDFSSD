'''
Micro Object Detector Net
the author:Luis
date : 11.25
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from models.base_models import vgg, vgg_base
from ptflops import get_model_complexity_info


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.in_channels = in_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MOD(nn.Module):
    def __init__(self, base, extras, upper, upper2, head, num_classes, size):
        super(MOD, self).__init__()
        self.num_classes = num_classes
        self.extras = nn.ModuleList(extras)
        self.size = size
        self.base = nn.ModuleList(base)
        # self.L2Norm = nn.ModuleList(extras)
        self.upper = nn.ModuleList(upper)
        self.upper2 = nn.ModuleList(upper2)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax()
        self.predict1 = nn.ModuleList(extra_predict1(self.size))
        self.predict2 = nn.ModuleList(extra_predict2(self.size))

    def forward(self, x, test=False):

        scale_source = []
        upper_source = []
        loc = []
        conf = []
        mid_trans = []
        # get the F.T of conv4
        for k in range(23):
            x = self.base[k](x)
        scale_source.append(x)
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        scale_source.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                scale_source.append(x)
        upper_source = scale_source
        lenscale = len(scale_source)
        orgin = x
        for k in range(len(self.upper) - 1):
            # bn = nn.BatchNorm2d(self.upper[lenscale-k-2].in_channels,affine=True)
            # print(self.upper[lenscale-k-2].in_channels)
            # print(self.upper[lenscale-k-1].out_channels)
            # print(scale_source[lenscale-k-2].size())
            se = SELayer(self.upper[lenscale - k - 1].out_channels, 16)
            upper_source[0] = upper_source[0] + se(self.upper[lenscale - k - 1](upper_source[lenscale - k - 1]))
            # upper_source[0] =upper_source[0]+  self.upper[lenscale-k-1](upper_source[lenscale-k-1])
        for k in range(len(self.upper) - 2):
            se = SELayer(self.upper2[lenscale - k - 1].out_channels, 16)
            upper_source[1] = upper_source[1] + se(self.upper2[lenscale - k - 1](upper_source[lenscale - k - 1]))
            # upper_source[1] = upper_source[1] + self.upper2[lenscale-k-1](upper_source[lenscale-k-1])
        bn = nn.BatchNorm2d(512, affine=True)
        upper_source[0] = bn(upper_source[0])
        # bn1 = nn.BatchNorm2d(1024,affine = True)
        # upper_source[1] = bn1(upper_source[1])

        predict_layer1 = []
        predict_layer1.append(upper_source[0])
        origin_fea = upper_source[0]
        # print('origin_fea',origin_fea.size())
        for k, v in enumerate(self.predict1):
            origin_fea = v(origin_fea)
            # print('ori',origin_fea.size())
            predict_layer1.append(origin_fea)

        bn = nn.BatchNorm2d(2048, affine=True)
        # print(predict_layer1[1].size())
        # print(upper_source[1].size())
        # predict_layer1[1] = bn(torch.cat([predict_layer1[1],upper_source[1]],1))
        predict_layer1[1] = predict_layer1[1] + upper_source[1]
        origin_fea2 = upper_source[1]
        for k, v in enumerate(self.predict2):
            origin_fea2 = v(origin_fea2)
            # predict_layer2.append(origin_fea2)
            # bn = nn.BatchNorm2d(v.out_channels*2,affine=True)
            # if not k==len(self.predict2)-1:
            #  predict_layer1[k+2] = bn(torch.cat([predict_layer1[k+2],origin_fea2],1))
            # else:
            # predict_layer1[k+2] = torch.cat([predict_layer1[k+2],origin_fea2],1)
            predict_layer1[k + 2] = predict_layer1[k + 2] + origin_fea2
        for (x, l, c) in zip(predict_layer1, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # for (x, l, c) in zip(upper_source, self.loc, self.conf):
        #    loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        #    conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print(loc.size())
        # print(conf.size())
        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
            # print(loc.size())
            # print(conf.size())
        return output


def low_pooling(vgg, extracts, size):
    if size == 300:
        up_size = layer_size('300')[k]
    elif size == 512:
        up_size = layer_size('512')[k]
    layers = []


def extra_predict1(size):
    if size == 300:
        layers = [BasicConv(512, 1024, kernel_size=3, stride=2, padding=1),
                  BasicConv(1024, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(512, 1024, kernel_size=3, stride=2, padding=1),
                  BasicConv(1024, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers


def extra_predict2(size):
    if size == 300:
        layers = [BasicConv(1024, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(1024, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers


def upper_deconv(vgg, extracts, size):
    layers = []
    layers2 = []
    if size == 300:
        layers.append(BasicConv(512, 128 * 4, kernel_size=1, padding=0))
        layers += [(BasicConv(vgg[-2].out_channels, 512, kernel_size=1, padding=0, up_size=38))]
        layers.append(BasicConv(extracts[1].out_channels, 512, kernel_size=1, padding=0, up_size=38))
        layers.append(BasicConv(extracts[3].out_channels, 512, kernel_size=1, padding=0, up_size=38))
        layers.append(BasicConv(extracts[5].out_channels, 512, kernel_size=1, padding=0, up_size=38))
        layers.append(BasicConv(extracts[7].out_channels, 512, kernel_size=1, padding=0, up_size=38))

        layers2.append(BasicConv(512, 128 * 4, kernel_size=1, padding=0))
        layers2 += [(BasicConv(vgg[-2].out_channels, 1024, kernel_size=1, padding=0, up_size=19))]
        layers2.append(BasicConv(extracts[1].out_channels, 1024, kernel_size=1, padding=0, up_size=19))
        layers2.append(BasicConv(extracts[3].out_channels, 1024, kernel_size=1, padding=0, up_size=19))
        layers2.append(BasicConv(extracts[5].out_channels, 1024, kernel_size=1, padding=0, up_size=19))
        layers2.append(BasicConv(extracts[7].out_channels, 1024, kernel_size=1, padding=0, up_size=19))

    elif size == 512:
        layers.append(BasicConv(512, 128 * 4, kernel_size=1, padding=0))
        layers.append(BasicConv(vgg[-2].out_channels, 512, kernel_size=1, padding=0, up_size=64))
        layers.append(BasicConv(extracts[1].out_channels, 512, kernel_size=1, padding=0, up_size=64))
        layers.append(BasicConv(extracts[3].out_channels, 512, kernel_size=1, padding=0, up_size=64))
        layers.append(BasicConv(extracts[5].out_channels, 512, kernel_size=1, padding=0, up_size=64))
        layers.append(BasicConv(extracts[7].out_channels, 512, kernel_size=1, padding=0, up_size=64))
        layers.append(BasicConv(extracts[9].out_channels, 512, kernel_size=1, padding=0, up_size=64))

        layers2.append(BasicConv(512, 128 * 4, kernel_size=1, padding=0))
        layers2.append(BasicConv(vgg[-2].out_channels, 1024, kernel_size=1, padding=0, up_size=32))
        layers2.append(BasicConv(extracts[1].out_channels, 1024, kernel_size=1, padding=0, up_size=32))
        layers2.append(BasicConv(extracts[3].out_channels, 1024, kernel_size=1, padding=0, up_size=32))
        layers2.append(BasicConv(extracts[5].out_channels, 1024, kernel_size=1, padding=0, up_size=32))
        layers2.append(BasicConv(extracts[7].out_channels, 1024, kernel_size=1, padding=0, up_size=32))
        layers2.append(BasicConv(extracts[9].out_channels, 1024, kernel_size=1, padding=0, up_size=32))

    return vgg, extracts, layers, layers2


def add_extras(cfg, i, batch_norm=False, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    # print(len(layers))
    return layers


def multibox(vgg, extra_layers, upper, upper2, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    loc_layers += [nn.Conv2d(upper[0].out_channels, cfg[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(upper[0].out_channels, cfg[0] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(upper):
        if k == 0:
            continue
        loc_layers += [nn.Conv2d(v.in_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.in_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    '''
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    '''
    return vgg, extra_layers, upper, upper2, (loc_layers, conf_layers)


layer_size = {
    '300': [38, 19, 10, 5, 3, 1],
    '512': [64, 32, 16, 8, 4, 2, 1],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return MOD(*multibox(*upper_deconv(vgg(vgg_base[str(size)], 3),
                                       add_extras(extras[str(size)], 1024, size=size), size),
                         mbox[str(size)], num_classes), num_classes=num_classes, size=size)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with torch.no_grad():
        model = build_net(size=512,num_classes=2)
        print(model)
        # x = torch.randn(16, 3, 300, 300)
        model.cuda()
        macs,params = get_model_complexity_info(model,(3,512,512),as_strings=True,print_per_layer_stat=True,verbose=True)
        print('MACs: {0}'.format(macs))
        print('Params: {0}'.format(params))