import os 
import numpy as np
from chainer.datasets import LabeledImageDataset
import chainer.iterators
from chainer.dataset import concat_examples
import chainer.dataset
from chainer import serializers
from chainer.cuda import to_cpu
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import chainer.datasets 
train,test=mnist.get_mnist(ndim=3)
train_iter=chainer.iterators.SerialIterator(train,1)
test_iter=chainer.iterators.SerialIterator(test,1)
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
    linear5=L.Linear(None,10)
)
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

models.to_gpu(0)
optimizer=optimizers.Adam()
optimizer.setup(models)
max_epoch=50
while train_iter.epoch<max_epoch:
    train_batch = train_iter.next()
    image_train, target_train = concat_examples(train_batch,0,padding=0)
    predicition_train=forward(image_train,models)
    loss = F.softmax_cross_entropy(predicition_train, target_train)
    models.cleargrads()
    loss.backward()
    optimizer.update()
    if train_iter.is_new_epoch:
        print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')
        test_losses = []
        test_accuracies = []


        while True:
            test_batch = t_batch = test_iter.next()
            image_test, target_test = concat_examples(test_batch,0,padding=0)
            prediction_test = forward(image_test,models)
            loss_test = F.softmax_cross_entropy(prediction_test, target_test)
            test_losses.append(to_cpu(loss_test.data))
            accuracy = F.accuracy(prediction_test, target_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.data)




            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break
        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(test_losses), np.mean(test_accuracies)))
        serializers.save_npz("./Model_check_point/resizebn_64_"+str(train_iter.epoch)+'.model',models)


