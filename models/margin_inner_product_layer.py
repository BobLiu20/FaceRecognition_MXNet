import sys
sys.path.insert(0, "/home/liubofang/bob_opensource/incubator-mxnet-v1.1.0/python")
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import math

class MarginInnerProduct(nn.Block):
    def __init__(self, margin_params, **kwargs):
        super(MarginInnerProduct, self).__init__(**kwargs)
        #  args
        self.in_units = margin_params.get("feature_dim", 512)
        self.out_units = margin_params["label_num"]
        #  lambda parameter
        self.lamb_iter = margin_params.get("lamb_iter", 0)
        self.lamb_base = margin_params.get("lamb_base", 1500)
        self.lamb_gamma = margin_params.get("lamb_gamma", 0.01)
        self.lamb_power = margin_params.get("lamb_power", 1)
        self.lamb_min = margin_params.get("lamb_min", 10)
        #  margin type
        self.margin = margin_params.get("margin", 4)
        self.margin_cos = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x
        ]
        #  training parameter
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(self.in_units, self.out_units))

    def forward(self, x, label):
        #  weight normalize
        with x.context:
            w = self.weight.data()
        with mx.autograd.pause():
            batch_size = x.shape[0]
            #  normalize weight
            with x.context:
                w[:] = mx.ndarray.L2Normalization(w, mode='instance')
        #  cos_theta = x'w/|x|. note: |w| = 1
        x_norm = nd.power(x, 2)
        x_norm = nd.sum(x_norm, axis=1)
        x_norm = nd.sqrt(x_norm)
        output = nd.dot(x, w)
        cos_theta = output / x_norm.reshape((-1, 1))
        cos_theta = nd.clip(cos_theta, -1, 1)
        #  cos_m_theta = cos(m * theta)
        cos_m_theta = self.margin_cos[self.margin](cos_theta)
        #  k
        with mx.autograd.pause():
            theta = nd.arccos(cos_theta)
            k = nd.sign((self.margin * theta / math.pi))
        #  i=j is phi_theta and i!=j is cos_theta
        phi_theta = ((-1)**k) * cos_m_theta - 2 * k
        x_norm_phi_theta = x_norm.reshape((-1, 1)) * phi_theta
        #  i=j index
        with mx.autograd.pause():
            index = nd.one_hot(label, output.shape[1])
        #  output
        with mx.autograd.pause():
            lamb = self.__get_lambda()
        output2 = output * (1.0 - index) + x_norm_phi_theta * index
        output3 = (output2 + lamb * output) / (1 + lamb)
        return output3

    def __get_lambda(self):
        self.lamb_iter += 1
        val = self.lamb_base * (1.0 + self.lamb_gamma * self.lamb_iter) ** (-self.lamb_power)
        val = max(self.lamb_min, val)
        if self.lamb_iter % 500 == 0:
            print ("Now lambda = {}".format(val))
        return val


if __name__ == "__main__":
    params = {"feature_dim": 4, "class_num": 6, 
              "lamb_iter":0,"lamb_base":1000,"lamb_gamma":0.0001,"lamb_power":1,"lamb_min":10}
    # ctx = [mx.gpu(4), mx.gpu(5)]
    ctx = [mx.gpu(1)]
    test = MarginInnerProduct(params)
    test.params
    test.initialize(ctx=ctx)
    x = nd.random.uniform(shape=(2,4))
    label = nd.zeros([2])
    x_list = mx.gluon.utils.split_and_load(x, ctx)
    label_list = mx.gluon.utils.split_and_load(label, ctx)
    with mx.autograd.record():
        for x, label in zip(x_list, label_list):
            y = test(x, label)
    print "y", y
    # print y
    y.backward()

