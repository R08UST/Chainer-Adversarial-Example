import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.datasets import LabeledImageDataset
import chainer.iterators
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from load_model import forward,load_model
from Attack_Function.fgsm import fgsm
models=load_model()
chainer.config.train=False #turn model to eval mode 

train,test=chainer.datasets.get_fashion_mnist(ndim=3)
train_iter=chainer.iterators.SerialIterator(test,1)





i=1
eps_list=[0.05,0.1,0.3,0.6]
for eps in eps_list:
    train_iter.current_position=0
    train_iter.epoch=0 
    train_iter.is_new_epoch = False
    train_iter._pushed_position = None #reset test_iter
    test_batch = train_iter.next()
    image_test, target_test = concat_examples(test_batch)
    image_test=F.resize_images(image_test,(32,32)).data
    adv=fgsm(models,forward,image_test,eps=eps)
    plt.subplot(1,4,i)
    plt.imshow(to_cpu(adv.squeeze()))
    plt.title("Label:{}".format(to_cpu(F.argmax(forward(adv,models)).data)))
    i=i+1
plt.savefig("demo.jpg")


