import sys
sys.path.insert(0, "/home/liubofang/bob_opensource/incubator-mxnet-v1.1.0/python")
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

class MarginInnerProduct(nn.Block):
    def __init__(self, margin_params, **kwargs):
        super(MarginInnerProduct, self).__init__(**kwargs)
        #  args
        self.ctx_list = []
        self.in_units = margin_params["in_units"]
        self.out_units = margin_params["out_units"]
        #  lambda parameter
        self.lamb_iter = margin_params["lamb_iter"]
        self.lamb_base = margin_params["lamb_base"]
        self.lamb_gamma = margin_params["lamb_gamma"]
        self.lamb_power = margin_params["lamb_power"]
        self.lamb_min = margin_params["lamb_min"]
        #  training parameter
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(self.out_units, self.in_units))
        #  global variable
        self.x_norm = []
        self.cos_theta = []
        self.sign_0 = []
        self.sign_3 = []
        self.sign_4 = []
        self.cos_theta_quadratic = []
        self.cos_theta_cubic = []
        self.cos_theta_quartic = []
        self.nd_1 = []
        self.nd_2 = []
        self.nd_3 = []
        self.nd_4 = []

    def forward(self, x, label):
        with mx.autograd.pause():
            batch_size = x.shape[0]
            ctx = x.context
            self.__init_all_tensor(batch_size, ctx)
            idx = self.ctx_list.index(ctx)
            _lambda = self.__get_lambda()
            #  normalize weight
            self.weight.data(ctx)[:] = mx.ndarray.L2Normalization(self.weight.data(ctx),
                                                                  mode='instance')
            #  x_norm = |x|
            for i in range(batch_size):
                nd.norm(x[i], out=self.x_norm[idx][i])
            #  cos_theta = x'w/|x|
            nd.dot(x, self.weight.data(ctx), False, True, self.cos_theta[idx])
            self.cos_theta[idx] /= self.x_norm[idx].reshape((-1, 1))
            #  sign_0 = sign(cos_theta)
            nd.sign(self.cos_theta[idx], out=self.sign_0[idx])

            ###  MarginInnerProductParameter_MarginType_QUADRUPLE
            #  cos_theta_quadratic && cos_theta_cubic && cos_theta_quartic
            nd.broadcast_power(self.cos_theta[idx], self.nd_2[idx],
                               out=self.cos_theta_quadratic[idx])
            # nd.broadcast_power(self.cos_theta[idx], self.nd_3[idx], out=self.cos_theta_cubic[idx])
            nd.broadcast_power(self.cos_theta[idx], self.nd_4[idx], out=self.cos_theta_quartic[idx])
            #  sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
            nd.broadcast_mul(self.cos_theta_quadratic[idx], self.nd_2[idx], out=self.sign_3[idx])
            nd.broadcast_sub(self.sign_3[idx], self.nd_1[idx], out=self.sign_3[idx])
            nd.sign(self.sign_3[idx], out=self.sign_3[idx])
            nd.broadcast_mul(self.sign_0[idx], self.sign_3[idx], out=self.sign_3[idx])
            #  sign_4 = 2 * sign_0 + sign_3 - 3
            nd.broadcast_mul(self.sign_0[idx], self.nd_2[idx], out=self.sign_4[idx])
            nd.broadcast_add(self.sign_4[idx], self.sign_3[idx], out=self.sign_4[idx])
            nd.broadcast_sub(self.sign_4[idx], self.nd_3[idx], out=self.sign_4[idx])
        #  Forward
        x = nd.dot(x, self.weight.data(ctx), False, True)
        #  + lambda * x'w
        x_tmp = x * _lambda
        #  |x| * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
        tmp = self.x_norm[idx].reshape((-1, 1)) * (self.sign_3[idx] * \
                (8.0 * self.cos_theta_quartic[idx] - 8.0 * self.cos_theta_quadratic[idx] + 1.0)\
                + self.sign_4[idx])
        oh = nd.one_hot(label, x.shape[1])
        x = x * (1.0 - oh) + tmp * oh
        x = x + x_tmp
        #  / (1 + lambda)
        x = x / (1 + _lambda)
        return x

    def __init_all_tensor(self, batch_size, ctx):
        if ctx in self.ctx_list:
            return
        self.ctx_list.append(ctx)
        self.x_norm.append(nd.zeros((batch_size,), ctx=ctx))
        self.cos_theta.append(nd.zeros((batch_size, self.out_units), ctx=ctx))
        self.sign_0.append(nd.zeros((batch_size, self.out_units), ctx=ctx))
        self.sign_3.append(nd.zeros((batch_size, self.out_units), ctx=ctx))
        self.sign_4.append(nd.zeros((batch_size, self.out_units), ctx=ctx))
        self.cos_theta_quadratic.append(nd.zeros((batch_size, self.out_units), ctx=ctx))
        self.cos_theta_cubic.append(nd.zeros((batch_size, self.out_units), ctx=ctx))
        self.cos_theta_quartic.append(nd.zeros((batch_size, self.out_units), ctx=ctx))
        self.nd_1.append(nd.array([1], ctx=ctx))
        self.nd_2.append(nd.array([2], ctx=ctx))
        self.nd_3.append(nd.array([3], ctx=ctx))
        self.nd_4.append(nd.array([4], ctx=ctx))

    def __get_lambda(self):
        self.lamb_iter += 1
        return self.lamb_base * (1.0 + self.lamb_gamma * self.lamb_iter) ** (-self.lamb_power)


if __name__ == "__main__":
    params = {"in_units": 4, "out_units": 6, 
              "lamb_iter":0,"lamb_base":1000,"lamb_gamma":0.0001,"lamb_power":1,"lamb_min":10}
    ctx = [mx.gpu(4), mx.gpu(5)]
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
    # y.backward()

