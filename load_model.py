import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.datasets import LabeledImageDataset
import chainer.iterators
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
def forward(data,model):
    h=F.max_pooling_2d(model.conv1(data),2,2)
    h=model.bn1(h)
    h=F.leaky_relu(model.conv2(h))
    h=model.bn2(h)
    h=F.max_pooling_2d(h,2,2)
    h=F.leaky_relu(model.conv3(h))
    h=model.bn3(h)
    h=F.leaky_relu(model.conv4(h))
    h=F.max_pooling_2d(h,2,2)
    h=model.bn4(h)
    h=F.leaky_relu(model.conv5(h))
    h=model.bn5(h)
    h=F.max_pooling_2d(h,2,2)
    y=model.linear5(h)
    return y
def load_model():
    models = chainer.Chain(
            conv1=L.Convolution2D(in_channels=None,out_channels=6,ksize=3,pad=1),
            conv2=L.Convolution2D(in_channels=None,out_channels=12,ksize=3,pad=1),
            conv3=L.Convolution2D(in_channels=None,out_channels=24,ksize=3,pad=1),
            conv4=L.Convolution2D(in_channels=None,out_channels=48,ksize=3,pad=1),
            conv5=L.Convolution2D(in_channels=None,out_channels=96,ksize=3,pad=1),
            bn1=L.BatchNormalization(6),
            bn2=L.BatchNormalization(12),
            bn3=L.BatchNormalization(24),
            bn4=L.BatchNormalization(48),
            bn5=L.BatchNormalization(96),
            linear5=L.Linear(None,10))
    serializers.load_npz('./Model_check_point/resizebn_64_36.model', models)
    #models.to_gpu(0)
    return models
