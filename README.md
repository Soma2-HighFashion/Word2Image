DCGAN.torch: Train your own image generator
[Original Source](https://github.com/soumith/dcgan.torch)
===========================================================

## BSD License
## For dcgan.torch software
## Copyright (c) 2015, Facebook, Inc. All rights reserved.

# Prerequisites
- Computer with Linux or OSX
- Torch-7
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.

# Installing dependencies
## Without GPU
- Install Torch:  http://torch.ch/docs/getting-started.html#_

## With NVIDIA GPU
- Install CUDA, and preferably CuDNN (optional).
  - Instructions for Ubuntu are here: [INSTALL.md](INSTALL.md)
- Install Torch:  http://torch.ch/docs/getting-started.html#_
- Optional, if you installed CuDNN, install cudnn bindings with `luarocks install cudnn`

## Display UI
Optionally, for displaying images during training and generation, we will use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

You can see training progress in your browser window. It will look something like this:
![display](https://github.com/soumith/dcgan.torch/raw/master/images/display_example.png "Example of display")


# 1. Train your own network

## 1.1 Train a generator on your own set of images.

### Preprocessing

- Create a folder called `myimages`.
- Inside that folder, create a folder called `images` and place all your images inside it.

### Training

```bash
DATA_ROOT=myimages dataset=folder th main.lua
```

## 1.2 All training options:

```lua
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,       
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 1,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
```

# 2. Use a pre-trained generator to generate images.
The generate script can operate in CPU or GPU mode.

to run it on the CPU, use:
```bash
gpu=0 net=[checkpoint-path] th generate.lua
```

for using a GPU, use:
```bash
gpu=1 net=[checkpoint-path] th generate.lua
```

##2.2. Generate large artsy images (tried up to 4096 x 4096 pixels)
```bash
gpu=0 batchSize=1 imsize=10 noisemode=linefull net=bedrooms_4_net_G.t7 th generate.lua
```

Controlling the `imsize` parameter will control the size of the output image.
Larger the imsize, larger the output image.

##2.3. Walk in the space of samples
```bash
gpu=0 batchSize=16 noisemode=line net=bedrooms_4_net_G.t7 th generate.lua
```

controlling the batchSize parameter changes how big of a step you take.
