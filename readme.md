# <div align = "center">EDREServer</div>

<a href="#Introduction-介绍">Introduction</a>

<a href="#Quickstart-启动项目">Quickstart</a>

<a href="#Project details-项目细节">Project details</a>

<a href="#For develop-开发">For develop</a>

## Introduction-介绍

本项目是基于Flask框架的web后端应用程序，实现了获取图像进行训练和渲染的功能，本文将介绍如何设置和运行这个程序。

![](./docs/images/image1.jpg)

本项目主要基于了[NerfStudio](https://github.com/nerfstudio-project/nerfstudio)项目的[2.0](https://github.com/nerfstudio-project/nerfstudio/tree/v0.2.0)版本进行实现，使用了其中nerfstudio以及scripts包，实现了获取前端请求以进行实现colmap，训练，渲染等功能。

对于数据库方面使用了Mysql数据库进行对每一个项目的管理，对文件包的管理。

## Quickstart-启动项目

#### 1.Installation-配置环境

##### 所需硬件基础

本后端需要英伟达显卡，同时需要安装CUDA，建议CUDA版本为11.3或者11.7

##### 创建环境

该项目所需`python>=3.7`，建议使用conda进行管理，基于conda创建环境如下：

```
conda create --name nsedre -y python=3.8
conda activate nsedre
python -m pip install --upgrade pip
```

##### 安装colmap

对于linux系统，安装的方法主要有以下3种：

***推荐**：1.使用colmap源码安装，前往[colmap](https://colmap.github.io/install.html#)获取详细步骤。

2.使用`sudo apt install colmap`进行安装

3.使用VKPG进行安装：

```
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap[cuda]:x64-linux
```

*上述方法都是在linux系统下进行安装，如需更多系统下的安装可前往[InstallingColmap](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html#installing-colmap)获取更多详细内容。

##### 安装torch和tiny-cuda-nn

For CUDA 11.3:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

For CUDA 11.7:

```
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

对于tiny-cuda-nn的安装，如果上述指令无法获取，则使用从[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/)仓库获取文件安装：

```
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn
cd bindings/torch
```

可选择使用以下几个命令进行安装：

```
python setup.py install
```

或者

```
pip install -e .
```

或者

```
pip install .
```

##### 安装其他依赖包

```
pip install -r requirements.txt
```

#### 2.Start-服务开启

运行一下命令实现服务的开启

```
python run.py
```

![image2](docs/images/image2.png)

出现该提示则说明服务成功运行

## Project details-项目细节



## For develop-开发