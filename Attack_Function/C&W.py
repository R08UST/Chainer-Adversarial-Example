import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.datasets import LabeledImageDataset
import chainer.iterators
from chainer.backends import cuda
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu,to_gpu
import cupy as np
from adam_variable import * # chainer do not support apply optim on variable,so implemented a function that can do it's job
from sklearn.preprocessing import OneHotEncoder

class L2:
    def __init__(self,model,forward_function,data,batch_size=32,confidence = 0.,
                 targeted = False, learning_rate = 1e-2,
                 binary_search_steps = 9, max_iterations = 10000,
                 abort_early = True,
                 initial_const =1e-3 ,
                 boxmin =0., boxmax = 1.):
        self.data=data
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.repeat = binary_search_steps >= 10
        self.model = model
        self.modify = chainer.Variable(np.zeros((self.batch_size,1,32,32),dtype=np.float32))  # modifier = tf.Variable(np.zeros(shape,dtype=np.float32))
        self.opt=chainer.optimizers.Adam(self.LEARNING_RATE)
        self.opt.setup(self.model) #   # Setup the adam optimizer and keep track of variables we're creating
        self.forward=forward_function
        self.const=chainer.Variable(np.zeros(self.batch_size,dtype=np.float32))
        self.enc=OneHotEncoder()
        self.enc.fit([[i] for i in range(25)])
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.Adam=Adam()
        self.Adam.init_state(self.modify)
    def attack_batch(self,imgs,targets):
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED and x.shape[0]>1:
                    x[y] -= self.CONFIDENCE
                elif self.TARGETED and x.shape[0]>1:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y
        self.modify = chainer.Variable(np.zeros((self.batch_size,1,32,32),dtype=np.float32))
        #preprocessing and calcaulate the const
        
        output = self.forward(imgs+self.modify, self.model)  # prediction BEFORE-SOFTMAX of the model
        real = F.sum(targets * output, 1)
        other = F.max((1. - targets) * output,1)  # must be one hot
        batch_size = self.batch_size
        # targets=chainer. #to one hot
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl2)
            # completely reset adam's internal state.

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                adv=F.clip(self.modify + imgs,0.,1.)
                #loss=F.softmax_cross_entropy(self.forward(adv,self.model),F.argmax(targets,1)) #auto gradiant check
                #continue
                l2dist = F.sum(F.square(adv -imgs), axis=(1, 2, 3))  # calculate L2 distance
                if self.TARGETED:
                    loss1 = F.maximum(np.zeros([self.batch_size],dtype=np.float32), other - real + self.CONFIDENCE)  # making the other class most likely
                else:
                    loss1 = F.maximum(np.zeros([self.batch_size],dtype=np.float32), real - other + self.CONFIDENCE)  # making this class least likely.
                loss2 = F.sum(l2dist)
                loss1 = F.sum(CONST*loss1)
                loss=loss2+loss1
                if iteration % (self.MAX_ITERATIONS // 10) == 0:# print out the losses every 10%
                    print(iteration, loss, loss1, loss2)
                self.model.cleargrads()
                self.modify.zerograd()
                loss.backward()
                
                self.Adam.update_core_cpu(self.modify)

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if loss.data > prev*.9999:
                        break
                    prev = loss.data
                # adjust the best result found so far
                for e,(l2,sc,ii) in enumerate(zip(l2dist,output,adv)):
                    if l2.data < bestl2[e] and compare(sc.data, np.argmax(targets[e])):
                        bestl2[e] = l2.data
                        bestscore[e] = np.argmax(sc.data)
                    if l2.data < o_bestl2[e] and compare(sc.data, np.argmax(targets[e])):
                        o_bestl2[e] = l2.data
                        o_bestscore[e] = np.argmax(sc.data)
                        o_bestattack[e] = ii.data

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(targets[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
    def attack(self,test):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        test_iter=chainer.iterators.SerialIterator(test,self.batch_size)
        r = []
        print('go up to',len(self.data))
        if test_iter.epoch<1:
            imgs = test_iter.next()
            imgs, targets = concat_examples(imgs,0,padding=0)
            imgs=F.resize_images(imgs,(32,32))
            imgs = imgs / 255  # rescale the image to [0,1]
            targets=F.argmax(self.forward(imgs,self.model),1).data
            targets=self.enc.transform(to_cpu(targets).reshape(-1,1)).toarray()
            targets=to_gpu(targets)
            r.extend(self.attack_batch(imgs, targets))
        return np.array(r)


