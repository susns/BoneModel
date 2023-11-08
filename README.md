# BoneModel

Super4PCS配置直接运行scripts/install.sh脚本，注意添加执行权限：
`chmod +x scripts/install.sh`
然后运行进行安装
`./scripts/install.sh`

如果安装失败，则参考官网安装方法：
https://storm-irit.github.io/OpenGR/a00002.html#library


## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `sap` using
```
conda env create -f environment.yaml
conda activate sap
```

Next, you should install [PyTorch3D](https://pytorch3d.org/) (**>=0.5**) yourself from the [official instruction](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md#3-install-wheels-for-linux).  

And install [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter):
```sh
conda install pytorch-scatter -c pyg
```
