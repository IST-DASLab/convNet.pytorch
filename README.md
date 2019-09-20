# Quantized horovod runs
1. Create virtual environment.
2. Install all required packages `pip install -r requirements.txt`.
3. Installing horovod. Go to horovod repo.
In horovod repo run
``` bash
export HOROVOD_NCCL_HOME=`path/to/nccl` (e.g.nccl//2.4.2-1/cuda10.0)
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_QUANTIZATION=1 \
pip -v install --ignore-installed .
```
4. In convNet repo:
Create directory `results` and run following command. 
Env variable HOROVOD_COMPRESSOR can be n(normalized), m(maxmin).

Env variable HOROVOD_QUANTIZATION_TYPE can be eL2, eLi, u(uniform).

`num_processes` is the number of gpus you want to run on. `save_dir` is directory inside dir results where graphs will be saved.
Directory with dataset is assumed to be inside directory ~/Datasets/.
``` bash
HOROVOD_QUANTIZATION_TYPE=L2 HOROVOD_COMPRESSOR=n HOROVOD_QUANTIZE_THRESHOLD=1 HOROVOD_QUANTIZE_BUCKET_SIZE=64 \ 
horovodrun -np $num_process -H localhost:$num_process python main.py --model resnet --horovod --model-config "{'depth': 18, 'regime': 'normal'}" \
--save $save_dir --dataset imagenet  --datasets-dir ~/Datasets/ --epochs 90 --quantization-bits 4 
```

To run on models from torchvision just put model's name(e.g. wide_resnet50_2) after --model keyword.
In this case you may need to define regime in --model-config argument. Example of argument:
```json
{
"regime": [
            {"epoch": 0, "optimizer": "SGD", "momentum": 0.9, 
             "lr": 0.1},
            {"epoch": 30, "lr": 1e-2},
            {"epoch": 60, "lr": 1e-3},
            {"epoch": 80, "lr": 1e-4}
            ]
}
```
# Convolutional networks using PyTorch

This is a complete training example for Deep Convolutional Networks on various datasets (ImageNet, Cifar10, Cifar100, MNIST).

Available models include:
```
'alexnet', 'amoebanet', 'darts', 'densenet', 'googlenet', 'inception_resnet_v2', 'inception_v2', 'mnist', 'mobilenet', 'mobilenet_v2', 'nasnet', 'resnet', 'resnet_se', 'resnet_zi', 'resnet_zi_se', 'resnext', 'resnext_se'
```

It is based off [imagenet example in pytorch](https://github.com/pytorch/examples/tree/master/imagenet) with helpful additions such as:
  - Training on several datasets other than imagenet
  - Complete logging of trained experiment
  - Graph visualization of the training/validation loss and accuracy
  - Definition of preprocessing and optimization regime for each model
  - Distributed training
 
 To clone:
 ```
 git clone --recursive https://github.com/eladhoffer/convNet.pytorch
 ```
 
 example for efficient multi-gpu training of resnet50 (4 gpus, label-smoothing, fast regime by fast-ai):
 ```
 python -m torch.distributed.launch --nproc_per_node=4  main.py --model resnet --model-config "{'depth': 50, 'regime': 'fast'}" --eval-batch-size 512 --save resnet50_fast --label-smoothing 0.1
```

This code can be used to implement several recent papers:
  - [Hoffer et al. (2018): Fix your classifier: the marginal value of training the last weight layer](https://arxiv.org/abs/1801.04540)
  - [Hoffer et al. (2018): Norm matters: efficient and accurate normalization schemes in deep networks](https://arxiv.org/abs/1803.01814)
  
      For example, training ResNet18 with L1 norm (instead of batch-norm):
      ```
      python main.py --model resnet --model-config "{'depth': 18, 'bn_norm': 'L1'}" --save resnet18_l1 -b 128
      ```
  - [Banner et al. (2018): Scalable Methods for 8-bit Training of Neural Networks](https://arxiv.org/abs/1805.11046)
  
    For example, training ResNet18 with 8-bit quantization:
    ```
    python main.py --model resnet --model-config "{'depth': 18, 'quantize':True}" --save resnet18_8bit -b 64
    ```
  - [Hoffer et al. (2019): Augment your batch: better training with larger batches](https://arxiv.org/abs/1901.09335)
    
    For example, training the resnet44 + cutout example in paper:
    ```
    python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"  --duplicates 40 --cutout -b 64 --epochs 100 --save resnet44_cutout_m-40
    ```
## Dependencies

- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization


## Data
- Configure your dataset path with ``datasets-dir`` argument
- To get the ILSVRC data, you should register on their site for access: <http://www.image-net.org/>


## Model configuration

Network model is defined by writing a <modelname>.py file in <code>models</code> folder, and selecting it using the <code>model</code> flag. Model function must be registered in <code>models/\_\_init\_\_.py</code>
The model function must return a trainable network. It can also specify additional training options such optimization regime (either a dictionary or a function), and input transform modifications.

e.g for a model definition:

```python
class Model(nn.Module):

    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.model = nn.Sequential(...)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 15, 'lr': 1e-3, 'weight_decay': 0}
        ]

        self.data_regime = [
            {'epoch': 0, 'input_size': 128, 'batch_size': 256},
            {'epoch': 15, 'input_size': 224, 'batch_size': 64}
        ]
    def forward(self, inputs):
        return self.model(inputs)
        
 def model(**kwargs):
        return Model()
```


# Citation

If you use the code in your paper, consider citing one of the implemented works.
```
@inproceedings{hoffer2018fix,
  title={Fix your classifier: the marginal value of training the last weight layer},
  author={Elad Hoffer and Itay Hubara and Daniel Soudry},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=S1Dh8Tg0-},
}
```
```
@inproceedings{hoffer2018norm,
  title={Norm matters: efficient and accurate normalization schemes in deep networks},
  author={Hoffer, Elad and Banner, Ron and Golan, Itay and Soudry, Daniel},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```
```
@inproceedings{banner2018scalable,
  title={Scalable Methods for 8-bit Training of Neural Networks},
  author={Banner, Ron and Hubara, Itay and Hoffer, Elad and Soudry, Daniel},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```
```
@article{hoffer2019augment,
  title={Augment your batch: better training with larger batches},
  author={Hoffer, Elad and Ben-Nun, Tal and Hubara, Itay and Giladi, Niv and Hoefler, Torsten and Soudry, Daniel},
  journal={arXiv preprint arXiv:1901.09335},
  year={2019}
}
```
