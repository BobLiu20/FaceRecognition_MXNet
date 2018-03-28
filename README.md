# Face Recognition

### Models
1. SphereFace with A-Softmax
 * Review paper: [link](https://arxiv.org/abs/1704.08063)
 * In this code, Two ways to implement a-softmax.
  * first: following the code of caffe what's wrote by paper author.
  * second: following paper in my understanding.
  * you can check it in ```models/margin_inner_product_layer.py```

### Environment
 * MxNet v1.1.0 (Gluon)

### Training
```
python training.py --input_paths=/your/data/path/10000_caffe_format.lst --working_root=/your/path/sphereface_mxnet --max_epoch=100 --img_size=112 --feature_dim=512 --label_num=10000 --process_num=15 --learning_rate=0.1 --model=SphereFaceNet --model_params='{"lamb_iter":0,"lamb_base":1000,"lamb_gamma":0.00001,"lamb_power":1,"lamb_min":500, "layer_type": "20layer"}' --try=0 --gpu_device=0,1,2,3 --batch_size=32
```

### Testing
 TODO

### Other DL framework
* The PyTorch implementation to see [here](https://github.com/BobLiu20/FaceRecognition_PyTorch)

### References
[sphereface](https://github.com/wy1iu/sphereface)   

