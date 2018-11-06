import chainer
import chainer.functions as F
import chainer.links as L
def fgsm(model,forward,image,eps=0.05,iter=1,clip_min=0.,clip_max=1.):
    targets = F.argmax(forward(image,model), axis=1)
    op=chainer.backends.cuda.get_array_module(image)
    eps=op.abs(eps)    
    for _ in range(iter):
        adv=chainer.Variable(image)
        loss=F.softmax_cross_entropy(forward(adv,model),targets)
        loss.backward()
        adv=adv.data+eps*op.sign(adv.grad)
        adv=op.clip(adv,clip_min,clip_max).astype(op.float32)
    return adv
