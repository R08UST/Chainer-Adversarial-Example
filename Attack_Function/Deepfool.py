import chainer
import chainer.functions as F
import chainer.links as L
def deepfool(images,model,forward,classes_num=10,overshot=0.02,max_iter=5):
    op=chainer.backends.cuda.get_array_module(images)
    labels=forward(images,model)
    input_shape=images.shape
    I = (labels.data).flatten().argsort()[::-1]
    I = I[0:classes_num]
    label = I[0]
    x=chainer.Variable(images)
    fs = forward(x,model)
    k_i = I[0]
    loop_i = 0
    w = op.zeros(input_shape)
    r_tot = op.zeros(input_shape)
    pert_image = images
    while k_i == label and loop_i < max_iter:
        pert = op.inf
        fs[0,I[0]].backward()
        grad_orig = x.grad
        for k in range(1, classes_num):
            x.cleargrad()
            fs[0, I[k]].backward()
            cur_grad = x.grad
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data
            pert_k = abs(f_k)/op.linalg.norm(w_k.flatten())
            if pert_k < pert:
                pert = pert_k
                w = w_k
        r_i =  (pert+1e-4) * w / op.linalg.norm(w)
        r_tot = F.cast(r_tot + r_i,op.float32)
        pert_image = images + (1+overshot)*r_tot
        x = chainer.Variable(pert_image.data)
        fs=forward(x,model)
        k_i=F.argmax(fs,1).data
        loop_i += 1
    r_tot = (1+overshot)*r_tot
    return r_tot, loop_i, label, k_i, pert_image
