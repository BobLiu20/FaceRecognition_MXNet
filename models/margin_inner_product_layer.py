import sys
sys.path.insert(0, "/home/liubofang/bob_opensource/incubator-mxnet-v1.1.0/python")
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

class MarginInnerProduct(nn.Block):
    def __init__(self, margin_params, **kwargs):
        super(MarginInnerProduct, self).__init__(**kwargs)
        #  args
        self.init_done = False
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

    def forward(self, x, label):
        batch_size = x.shape[0]
        with mx.autograd.pause():
            self.__init_all_tensor(batch_size, x.context)
            _lambda = self.__get_lambda()
            #  normalize weight
            self.weight.set_data(mx.ndarray.L2Normalization(self.weight.data(), mode='instance'))
            #  x_norm = |x|
            for i in range(batch_size):
                nd.norm(x[i], out=self.x_norm[i])
            #  cos_theta = x'w/|x|
            nd.dot(x, self.weight.data(), False, True, self.cos_theta)
            self.cos_theta /= self.x_norm.reshape((-1, 1))
            #  sign_0 = sign(cos_theta)
            nd.sign(self.cos_theta, out=self.sign_0)

            ###  MarginInnerProductParameter_MarginType_QUADRUPLE
            #  cos_theta_quadratic && cos_theta_cubic && cos_theta_quartic
            nd.broadcast_power(self.cos_theta, self.nd_2, out=self.cos_theta_quadratic)
            nd.broadcast_power(self.cos_theta, self.nd_3, out=self.cos_theta_cubic)
            nd.broadcast_power(self.cos_theta, self.nd_4, out=self.cos_theta_quartic)
            #  sign_3 = sign_0 * sign(2 * cos_theta_quadratic_ - 1)
            nd.broadcast_mul(self.cos_theta_quadratic, self.nd_2, out=self.sign_3)
            nd.broadcast_sub(self.sign_3, self.nd_1, out=self.sign_3)
            nd.sign(self.sign_3, out=self.sign_3)
            nd.broadcast_mul(self.sign_0, self.sign_3, out=self.sign_3)
            #  sign_4 = 2 * sign_0 + sign_3 - 3
            nd.broadcast_mul(self.sign_0, self.nd_2, out=self.sign_4)
            nd.broadcast_add(self.sign_4, self.sign_3, out=self.sign_4)
            nd.broadcast_sub(self.sign_4, self.nd_3, out=self.sign_4)
        #  Forward
        x = nd.dot(x, self.weight.data(), False, True)
        #  + lambda * x'w
        x_tmp = x * _lambda
        #  |x| * (sign_3 * (8 * cos_theta_quartic - 8 * cos_theta_quadratic + 1) + sign_4)
        tmp = self.x_norm.reshape((-1, 1)) * (self.sign_3 * \
                (8.0 * self.cos_theta_quartic - 8.0 * self.cos_theta_quadratic + 1.0)\
                + self.sign_4)
        oh = nd.one_hot(label, x.shape[1])
        x = x * (1.0 - oh) + tmp * oh
        x = x + x_tmp
        #  / (1 + lambda)
        x = x / (1 + _lambda)
        return x

    def __init_all_tensor(self, batch_size, ctx):
        if self.init_done == False:
            self.init_done = True
            self.x_norm = nd.zeros((batch_size,), ctx=ctx)
            self.cos_theta = nd.zeros((batch_size, self.out_units), ctx=ctx)
            self.sign_0 = nd.zeros((batch_size, self.out_units), ctx=ctx)
            self.sign_3 = nd.zeros((batch_size, self.out_units), ctx=ctx)
            self.sign_4 = nd.zeros((batch_size, self.out_units), ctx=ctx)
            self.cos_theta_quadratic = nd.zeros((batch_size, self.out_units), ctx=ctx)
            self.cos_theta_cubic = nd.zeros((batch_size, self.out_units), ctx=ctx)
            self.cos_theta_quartic = nd.zeros((batch_size, self.out_units), ctx=ctx)
            self.nd_1 = nd.array([1], ctx=ctx)
            self.nd_2 = nd.array([2], ctx=ctx)
            self.nd_3 = nd.array([3], ctx=ctx)
            self.nd_4 = nd.array([4], ctx=ctx)
            # self.output_data = nd.zeros((batch_size, self.out_units), ctx=ctx)
            # self.output_data_tmp = nd.zeros((batch_size, self.out_units), ctx=ctx)

    def __get_lambda(self):
        self.lamb_iter += 1
        return self.lamb_base * (1.0 + self.lamb_gamma * self.lamb_iter) ** (-self.lamb_power)


if __name__ == "__main__":
    params = {"in_units": 4, "out_units": 6, 
              "lamb_iter":0,"lamb_base":1000,"lamb_gamma":0.0001,"lamb_power":1,"lamb_min":10}
    test = MarginInnerProduct(params)
    test.params
    test.initialize(ctx=mx.gpu())
    x = nd.random.uniform(shape=(2,4), ctx=mx.gpu())
    label = nd.zeros([2], ctx=mx.gpu())
    with mx.autograd.record():
        y = test(x, label)
    print y
    y.backward()

