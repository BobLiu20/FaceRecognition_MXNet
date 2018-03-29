import sys
sys.path.insert(0, "/home/liubofang/bob_opensource/incubator-mxnet-v1.1.0/python")
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, loss
from mxnet import gluon

from margin_inner_product_layer import MarginInnerProduct

class PReLU(gluon.HybridBlock):
    def __init__(self, alpha=0.25, **kwargs):
        super(PReLU, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get('alpha', shape=(1,), init=mx.init.Constant(alpha))

    def hybrid_forward(self, F, x, alpha):
        pos = F.relu(x)
        neg = F.broadcast_mul(F.negative(alpha), F.relu(F.negative(x)))
        return pos + neg

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__)

class CNNResidualBlock(nn.Block):
    def __init__(self, num_output_cnn, num_output_res, num_residual, **kwargs):
        super(CNNResidualBlock, self).__init__(**kwargs)
        self.num_residual = num_residual
        self.cnn = nn.Conv2D(num_output_cnn, kernel_size=3, strides=2, padding=1)
        self.prelu = PReLU()
        for i in range(self.num_residual):
            setattr(self, 'residual%d' % i, nn.Sequential())
            getattr(self, 'residual%d' % i).add(
                nn.Conv2D(num_output_res, kernel_size=3, strides=1, padding=1),
                PReLU(),
                nn.Conv2D(num_output_res, kernel_size=3, strides=1, padding=1),
                PReLU()
            )

    def forward(self, x):
        x = self.cnn(x)
        x = self.prelu(x)
        for i in range(self.num_residual):
            _x = getattr(self, 'residual%d' % i)(x)
            x = x + _x
        return x

class SphereFaceNet(gluon.Block):
    def __init__(self, model_params, **kwargs):
        super(SphereFaceNet, self).__init__(**kwargs)
        with self.name_scope():
            label_num  = model_params["label_num"]
            self.models = self._main_net(model_params["layer_type"], model_params["feature_dim"],
                                         label_num)
            #  A-softmax = margin_inner_product + softmax_loss
            self.margin_inner_product = MarginInnerProduct(model_params,
                                                           prefix="MarginFC%d"%(label_num))
            self.softmax_loss = loss.SoftmaxCrossEntropyLoss()

    def forward(self, x, label):
        x = self.models(x)
        x = self.margin_inner_product(x, label)
        return self.softmax_loss(x, label)

    def _main_net(self, layer_type, feature_dim, label_num):
        model = nn.Sequential()
        if layer_type == "20layer":
            model.add(
                CNNResidualBlock(64, 64, 1),
                CNNResidualBlock(128, 128, 2),
                CNNResidualBlock(256, 256, 4),
                CNNResidualBlock(512, 512, 1)
            )
        else:
            raise Exception("Unsupport layer type.")
        model.add(nn.Dense(feature_dim))
        return model


if __name__ == "__main__":
    margin_params = {"feature_dim": 512, "label_num": 6, "lamb_iter": 0, "lamb_base": 1000,
                     "lamb_gamma": 0.12, "lamb_power": 1, "lamb_min": 10}
    margin_params["layer_type"] = "20layer"
    test = SphereFaceNet(512, 6, margin_params)
    print test
    test.initialize(ctx=mx.gpu())
    x = nd.random.uniform(shape=(2, 3, 112, 112), ctx=mx.gpu())
    label = nd.zeros([2], ctx=mx.gpu())
    print test(x, label)

