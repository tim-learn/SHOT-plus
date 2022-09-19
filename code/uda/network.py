import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
# from efficientnet_pytorch import EfficientNet
# from eff_utils import (
#     round_filters,
#     round_repeats,
#     drop_connect,
#     get_same_padding_conv2d,
#     get_model_params,
#     efficientnet_params,
#     load_pretrained_weights,
#     Swish,
#     MemoryEfficientSwish,
# )

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

# For SVHN dataset
class DTN(nn.Module):
    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )   
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout()
                )
        self.classifier = nn.Linear(512, 10)
        self.__in_features = 512

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.fc_params = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, 10)
        self.__in_features = 500

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter=10000):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    # x = self.dropout1(x)
    # x = self.ad_layer2(x)
    # x = self.relu2(x)
    # x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


class ssl_classifier(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(ssl_classifier, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    # self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 4)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    # x = self.ad_layer2(x)
    # x = self.relu2(x)
    # x = self.dropout2(x)
    y = self.ad_layer3(x)
    # y = self.sigmoid(y)
    return y

class Identity(nn.Module):

    def __init__(self, sub=1.0):
        super(Identity, self).__init__()
        self.sub=sub

    def forward(self,x):
        return x*self.sub

class AlexNet_BVLC(nn.Module):
    def __init__(self, dropout=False):
        super(AlexNet_BVLC, self).__init__()
        self.features = nn.Sequential(OrderedDict([
		 ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
		 ("relu1", nn.ReLU(inplace=True)),
		 ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
		 ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
		 ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
		 ("relu2", nn.ReLU(inplace=True)),
		 ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
		 ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
		 ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
		 ("relu3", nn.ReLU(inplace=True)),
		 ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
		 ("relu4", nn.ReLU(inplace=True)),
		 ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
		 ("relu5", nn.ReLU(inplace=True)),
		 ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
	    ]))
        self.classifier = nn.Sequential(OrderedDict([
		 ("fc6", nn.Linear(256 * 6 * 6, 4096)),
		 ("relu6", nn.ReLU(inplace=True)),
		 ("drop6", nn.Dropout() if dropout else Identity(sub=0.5)),
		 ("fc7", nn.Linear(4096, 4096)),
		 ("relu7", nn.ReLU(inplace=True)),
		 ("drop7", nn.Dropout() if dropout else Identity(sub=0.5)),
		 # ("fc8", nn.Linear(4096, 1000))
	    ]))

class DecafBase(nn.Module):
  def __init__(self):
    super(DecafBase, self).__init__()
    model = AlexNet_BVLC()
    model_old = torch.load('/Checkpoint/liangjian/tran/icml20/alexnet_caffe.pth.tar')
    st = model.state_dict()
    model_dict = {k:v for k, v in model_old.items() if k in st.keys()}
    
    # model_old = torch.load('./caffe_places205.dat')['state_dict']    
    # model_dict = {}
    # for k, v in model_old.items():
    #     if k.__contains__('fc'):
    #         model_dict['classifier.'+k] = v
    #     else:
    #         model_dict['features.'+k] = v
    # # pdb.set_trace()
    # model_dict = {k:v for k, v in model_dict.items() if k in st.keys()}
    st.update(model_dict)
    model.load_state_dict(st)

    self.features = model.features
    self.classifier = model.classifier
    self.in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = '/Checkpoint/liangjian/tran/icml20/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model

class AlexBase(nn.Module):
  def __init__(self):
    super(AlexBase, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    # model_alexnet = alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.in_features = model_alexnet.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResSSLBase(nn.Module):
    def __init__(self):
        super(ResSSLBase, self).__init__()

        model_resnet = models.resnet50()
        # model_old = torch.load('in22k_jigsaw.dat')['state_dict']
        model_old = torch.load('/Checkpoint/liangjian/tran/icml20/places205.dat')['state_dict']
        st = model_resnet.state_dict()
        model_dict = {k:v for k, v in model_old.items() if k in st.keys()}
        st.update(model_dict)
        model_resnet.load_state_dict(st)

        # model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class EfficientBase(nn.Module):
    def __init__(self, eff_name='efficientnet-b0'):
        super(EfficientBase, self).__init__()
        model = EfficientNet.from_pretrained(eff_name)
        # pdb.set_trace()
        # self.extract_features = model.extract_features
        self._swish = MemoryEfficientSwish()
        self._conv_stem = model._conv_stem
        self._bn0 = model._bn0
        self._blocks = model._blocks
        self._global_params = model._global_params
        self._bn1 = model._bn1
        self._conv_head = model._conv_head

        self._avg_pooling = model._avg_pooling
        self._dropout = model._dropout
        self.in_features = model._fc.in_features

    def forward(self, x):
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self._dropout(x)
        return x

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn" or self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.bn(x)
        if self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.relu(x)
        if self.type == "bn_relu_drop":
            x = self.dropout(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            
            x = torch.nn.functional.normalize(x, dim=1, p=2)
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)
        return x

class feat_classifier2(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256):
        super(feat_classifier2, self).__init__()
        self.fc1 = nn.Linear(bottleneck_dim, bottleneck_dim//2)
        self.fc1.apply(init_weights)
        self.fc2 = nn.Linear(bottleneck_dim//2, class_num)
        self.fc2.apply(init_weights)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# class feat_classifier_two(nn.Module):
#     def __init__(self, class_num, input_dim, bottleneck_dim=256):
#         super(feat_classifier_two, self).__init__()
#         self.type = type
#         self.fc0 = nn.Linear(input_dim, bottleneck_dim)
#         self.fc0.apply(init_weights)
#         self.fc1 = nn.Linear(bottleneck_dim, class_num)
#         self.fc1.apply(init_weights)

#     def forward(self, x):
#         x = self.fc0(x)
#         x = self.fc1(x)
#         return x

class feat_classifier_twin(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier_twin, self).__init__()
        self.type = type
        # if type == 'wn':
        #     self.fc1 = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        #     self.fc2 = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        #     self.fc1.apply(init_weights)
        #     self.fc2.apply(init_weights)
        # else:
        #     self.fc1 = nn.Linear(bottleneck_dim, class_num)
        #     self.fc2 = nn.Linear(bottleneck_dim, class_num)
        #     self.fc1.apply(init_weights)
        #     self.fc2.apply(init_weights)

        # self.fc0_a = nn.Linear(bottleneck_dim, bottleneck_dim)
        # self.fc0_b = nn.Linear(bottleneck_dim, bottleneck_dim)

        # self.relu_a = nn.ReLU(inplace=True)
        # self.relu_b = nn.ReLU(inplace=True)

        # self.dropout_a = nn.Dropout(0.5)
        # self.dropout_b = nn.Dropout(0.5)

        if type == 'wn':
            self.fc_a = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc_b = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc_a = nn.Linear(bottleneck_dim, class_num)
            self.fc_b = nn.Linear(bottleneck_dim, class_num)

        self.fc_a.apply(init_weights)
        self.fc_b.apply(init_weights)

    def forward(self, x):
        # x1 = self.fc0_a(x)
        # x1 = self.relu_a(self.dropout_a(x1))
        f1 = self.fc_a(x)
        # x2 = self.fc0_b(x)
        # x2 = self.relu_b(self.dropout_b(x2))
        f2 = self.fc_b(x)
        return f1, f2

class rot_classifier(nn.Module):
    def __init__(self, class_num, feature_dim=2048, bottleneck_dim=256, type="linear"):
        super(rot_classifier, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(feature_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        # x = self.relu(x)
        x = self.fc(x)
        return x


class feat_classifier_nonlinear(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier_nonlinear, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        # self.relu = nn.ReLU(inplace=True)
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc0(x)
        # x = self.relu(x)
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            
            x = torch.nn.functional.normalize(x, dim=1, p=2)
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)
        return x

class feat_cosine_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="cosine"):
        super(feat_cosine_classifier, self).__init__()
        self.type = type
        self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        if self.type in {'cosine'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            x = torch.nn.functional.normalize(x, dim=1, p=2)
            # pdb.set_trace()
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)
        return x


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )   
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x