Deep learning tests
===================

Sketchbook for developing a simple binary classifier in Python using deep learning techniques such as multilayer perceptrons and convolutional neural networks. Includes a number of tutorials and examples, mostly to get the hang of various libraries available for the task.

Installation
------------

There are two ways to run these programs: using CPUs (slow) or GPUs (fast). The Python setup for both is the same (assumes you're using virtualenvs and have virtualenvwrapper installed): 

```
mkvirtualenv deeplearn
pip install -r requirements.txt
```

To verify that everything worked, run:

```
python tutorials/mnist-keras-mlp.py
```

This will build and train a simple multilayer perceptron using your CPU.

### Optional: Installing matplotlib on OSX

If you want to be able to visualize the output of the network, you'll also want to set up [matplotlib](http://matplotlib.org/) to be able to open a plotting window on your Mac:

```
echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc
```

### Optional: Configuring for GPU access using CUDA
 
GPU setup is optional but worthwhile. And it comes with a catch: namely that it won't work on a Mac unless you have an NVIDIA graphics card (which you probably don't). The libraries we're using here, namely Theano, rely on [CUDA](https://en.wikipedia.org/wiki/CUDA), which is an NVIDIA-specific parallel processing platform. That means your best option for using a GPU is to boot up an EC2 instance with NVIDIA GPUs.

This should be pretty painless, so long as you have a key to set up servers in Amazon's EC2 service. First, start a GPU instance on EC2. This takes a few minutes.

First you may want to set some environment variables, described here:

**Required:**

  - `GPU_INSTANCE_KEY`: The name of your Amazon keyfile, minus the .pem extension.

**Optional:**

  - `GPU_INSTANCE_REGION`: The region in which your GPU lives. Defaults to `us-west-1`.
  - `GPU_INSTANCE_AMI_ID`: The AMI ID you want to use. Defaults to community AMI `ami-91b077d5`.
  - `GPU_INSTANCE_TYPE`: The type of EC2 instance. Defaults to `g2.2xlarge`.
  - `GPU_INSTANCE_NAME`: The name of your instance. Defaults to `deeplearn-gpu`.

Once that's set up, you can create and bootstrap an instance by typing:

```
fab gpu_up
```

Pay attention and follow the prompts. After that's done, SSH into the instance using your key:

```
ssh -i ~/path/to/your_key.pem ubuntu@gpu-server-hostname-from-previous-step
```

You can also try using `fab gpu_go`. Now try running some code from the remote server. This app should live in ~/deeplearn-tests.

```
cd ~/deeplearn-test/tutorials
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist-keras-cnn.py
```

If that works, you're good to go. Don't forget to shut down the instance once you're done, unless you like spending money:

```
fab gpu_down
```

Resources
---------

The libraries we're using for these tests include:

  - [Theano](http://deeplearning.net/software/theano/): A mathematical library that allows us to perform calculations on GPUs (among other things) and underpins most Python deep learning systems.
  - [Keras](http://keras.io/): Among the more mature and user-friendly libraries for building neural networks in Python. Requires Theano.

Questions?
----------

Contact Chase: chase.davis@gmail.com