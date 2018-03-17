# coding='utf-8'
import os
import sys
import argparse
import numpy as np
import time
import datetime
import json

sys.path.insert(0, "/home/liubofang/bob_opensource/incubator-mxnet-v1.1.0/python")
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'models'))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'common'))
from batch_reader import BatchReader
import models

def train(prefix, **arg_dict):
    batch_size = arg_dict['batch_size']
    img_size = arg_dict['img_size']
    # batch generator
    _batch_reader = BatchReader(**arg_dict)
    _batch_generator = _batch_reader.batch_generator()
    # net
    net =  models.init(arg_dict["model"], feature_dim=arg_dict["feature_dim"],
                       label_num=arg_dict["label_num"],
                       model_params=json.loads(arg_dict["model_params"]))
    if arg_dict["restore_ckpt"]:
        print "resotre checkpoint from %s" % (arg_dict["restore_ckpt"])
        net.load_params(arg_dict['restore_ckpt'], ctx=mx.gpu())
    else:
        net.initialize(init=mx.init.Xavier(), ctx=mx.gpu())
    print (net)
    # trainer
    trainer = gluon.Trainer(net.collect_params(), "sgd", # adam
                            {"learning_rate": arg_dict['learning_rate']})
    # start loop
    print ("Start to training...")
    start_time = time.time()
    step = 0
    display = 100
    loss_list = []
    while not _batch_reader.should_stop():
        batch = _batch_generator.next()
        image = nd.array(batch[0], ctx=mx.gpu(), dtype='float32')
        label = nd.array(batch[1], ctx=mx.gpu(), dtype='float32')
        #  normalization. keep in-place operation
        image = nd.transpose(image, (0,3,1,2))
        image -= 127.5
        image *= 0.0078125
        with autograd.record():
            loss = net(image, label)
        loss.backward()
        trainer.step(batch_size)
        nd.waitall()
        loss_list.append(nd.mean(loss).asscalar())
        if step % display == 0:
            end_time = time.time()
            cost_time, start_time = end_time - start_time, end_time
            sample_per_sec = int(display * batch_size / cost_time)
            sec_per_step = cost_time / float(display)
            loss_display = "[loss: %.5f]" % (np.mean(loss_list))
            print ('[%s] epochs: %d, step: %d, lr: %.5f, loss: %s,'\
                   'sample/s: %d, sec/step: %.3f' % (
                   datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 
                   _batch_reader.get_epoch(), step, trainer.learning_rate, loss_display,
                   sample_per_sec, sec_per_step))
            landmark_loss_list, angle_loss_list = [], []
        if step % 50000 == 0:
            # change lr
            trainer.set_learning_rate(trainer.learning_rate * 0.95)
            print ("change lr to %f" % trainer.learning_rate)
            # save checkpoint
            checkpoint_path = os.path.join(prefix, 'model.params')
            net.save_params(checkpoint_path)
            print ("save checkpoint to %s" % checkpoint_path)
        step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_paths', type=str, nargs='+', default='')
    parser.add_argument('--working_root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--max_epoch', type=int, default=100000, help="Training will be stoped in this case.")
    parser.add_argument('--img_size', type=int, default=128, help="The size of input for model")
    parser.add_argument('--feature_dim', type=int, default=512, help="dim of face feature")
    parser.add_argument('--label_num', type=int, default=696877, help="the label num of your training set")
    parser.add_argument('--process_num', type=int, default=20, help="The number of process to preprocess image.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="lr")
    parser.add_argument('--model', type=str, default='SphereFaceNet', help="Model name. Check models.py")
    parser.add_argument('--model_params', type=str, default='{}', help="params for model. dict format")
    parser.add_argument('--restore_ckpt', type=str, help="Resume training from special ckpt.")
    parser.add_argument('--try', type=int, default=0, help="Saving path index")
    parser.add_argument('--gpu_device', type=str, default='7', help="GPU index")
    arg_dict = vars(parser.parse_args())
    prefix = '%s/%s/dim%d_size%d_try%d' % (
        arg_dict['working_root'], arg_dict['model'], 
        arg_dict["feature_dim"], arg_dict['img_size'], arg_dict['try'])
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # set up environment
    os.environ['CUDA_VISIBLE_DEVICES']=arg_dict['gpu_device']

    train(prefix, **arg_dict)

if __name__ == "__main__":
    main()

