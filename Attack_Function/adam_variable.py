from __future__ import division
import math

import numpy

from chainer.backends import cuda
from chainer import optimizer


_default_hyperparam_alpha = 0.001
_default_hyperparam_beta1 = 0.9
_default_hyperparam_beta2 = 0.999
_default_hyperparam_eps = 1e-8
_default_hyperparam_eta = 1.0
_default_hyperparam_weight_decay_rate = 0
_default_hyperparam_amsgrad = False
_default_hyperparam_lr=0.01



class Adam():
    def __init__(self,
                lr=_default_hyperparam_lr,
                 alpha=_default_hyperparam_alpha,
                 beta1=_default_hyperparam_beta1,
                 beta2=_default_hyperparam_beta2,
                 eps=_default_hyperparam_eps,
                 eta=_default_hyperparam_eta,
                 weight_decay_rate=_default_hyperparam_weight_decay_rate,
                 amsgrad=_default_hyperparam_amsgrad):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eta = eta
        self.weight_decay_rate = weight_decay_rate
        self.amsgrad = amsgrad
        self.lr=lr
        self.state={}
    def init_state(self,param):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)
            if self.amsgrad:
                self.state['vhat'] = xp.zeros_like(param.data)
    def update_core_cpu(self,param):
        xp = cuda.get_array_module(param.data)
        grad = param.grad
        if grad is None:
            return
        eps = grad.dtype.type(self.eps)
        if self.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Adam optimizer is too small for {} ({})'.format(
                    grad.dtype.name, self.eps))
        m, v = self.state['m'], self.state['v']

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)

        if self.amsgrad:
            vhat = self.state['vhat']
            xp.maximum(vhat, v, out=vhat)
        else:
            vhat = v
        param.data -= self.eta * (self.lr * m / (xp.sqrt(vhat) + self.eps) +
                                self.weight_decay_rate * param.data)