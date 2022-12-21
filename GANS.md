<div style="background-color: #ccffcc; padding: 10px;">
    <h1> Tutorial 5 </h1>
    <h2> Introduction to Generative Adversarial Networks </h2>
</div>      


# Overview

This Jupyter notebook will walk you through building a simple GAN model and training it on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) using [PyTorch](https://pytorch.org/). PyTorch is a very powerful library written in Python and C++ that facilitates the building and training of neural networks.

This tutorial was adapted from Caitlin Howarth's Presentation and tutorial as part of the LIFD Work shop series. Some of the code and explanations from this tutorial came from [Aleksa Gordić's pyctorch-gans repository](https://github.com/gordicaleksa/pytorch-gans). It contains some more in-depth explanations, as well as examples of more advanced GAN models than we covered here.


## The very basics

If you know nothing about neural networks there is a [toy neural network python code example](https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS/tree/main/ToyNeuralNetwork) included in the [LIFD ENV ML Notebooks Repository]( https://github.com/cemac/LIFD_ENV_ML_NOTEBOOKS). Creating a 2 layer neural network to illustrate the fundamentals of how Neural Networks work and the equivlent code using the python machine learning library [keras](https://keras.io/).

## Recommended reading


* [Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
* [Introduction to Generative Adversarial Newtworks](https://towardsdatascience.com/fundamentals-of-generative-adversarial-networks-b7ca8c34f0bc)
* [Auto Encoders](https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798)
* [Neural Networks for Regression Problems](https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33)
* [Digitizing Sketches of the Earth’s Surface with the AI Behind Deep Fakes (IBM Earth Science Example)](https://www.ibm.com/blogs/research/2019/06/good-gans/)

<hr>


<div style="background-color: #e6ccff; padding: 10px;">

<h1> Machine Learning Theory </h1>

Generative Adversarial Newtworks are a pair of Neural Networks that work to generate rather than predict. The example you've probably see in [mainstream media](https://www.artificialintelligence-news.com/2022/05/09/kendrick-lamar-uses-deepfakes-in-latest-music-video/) is deepfakes, where the network has been trained to produce images of humans that are so good even humans can't tell the difference  

Generative Adversarial Newtworks are made up of two [Neural Networks](): a generator and a discriminator. Both networks are equally bad at their respective tasks initially. As training continues, both networks compete with one another to get better at their respective tasks. Eventually, the generator becomes so good that humans can’t tell the difference (hopefully!)

**The Generator**

<a href="">
<img src="figs/generator.png">
</a>

The generator takes a latent code as an input, and then decodes that vector into an output.

Different latent codes should result in different outputs.

In the original implementation of GANs, the generator never actually sees the training dataset, and is solely guided by the discriminator.


**The Discriminator**    
<a href="">
<img src="figs/discriminator.png">
</a>


The discriminator is just a simple regression network and outputs a score from 0 (fake) to 1 (real).

Values closer to 0 or 1 represent increased certainty that a sample is fake or real respectively.

**Loss Functions**

Adversarial loss is essentially a minimax game, as proposed by Ian Goodfellow et al. [1]

The discriminator attempts to maximise the likelihood of correctly predicting real samples as real, and fake samples as fake.

The generator attempts to maximise the likelihood of the discriminator predicting that the fake samples it generates are real.

* The discriminator loss is defined as: , where  is the discriminator,  is the generator,  is a sample from the real distribution and  is a latent vector given to the generator.

* The generator loss is defined as: , where  is the discriminator,  is the generator, and  is a latent vector given to the generator.

References:

* [1] Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. 2014. Generative Adversarial Networks. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)  [cs, stat].

</div>    


<div style="background-color: #cce5ff; padding: 10px;">

<h1> Python </h1>

Basic python knowledge is assumed for this tutorial. For this tutorial the main machine learning library we'll be working [PyTorch](https://pytorch.org/). Python specific information will be colour coded blue.


## PyTorch

[PyTorch](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e) is an open source deep learning framework Pytorch is very *pythonic*, so if you are used to coding in python using PyTorch should come naturally.


References:

* [https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
* [https://www.nvidia.com/en-us/glossary/data-science/pytorch/](https://www.nvidia.com/en-us/glossary/data-science/pytorch/)    

</div>

<hr>



<div style="background-color: #ffffcc; padding: 10px;">

<h1> Requirements </h1>

These notebooks should run

<h2> Python Packages: </h2>

* Python 3
* PyTorch
* time
* os
* numpy
* matplotlib=3.0


<h2> Data Requirements</h2>

This notebook referes to some data included in the git hub repositroy

</div>

**Contents:**

1. [Download MNIST Dataset](#Download-MNIST-Dataset)
2. [Building the GAN](#Building-the-GAN)
3. [Preparing to train](#Preparing-to-train)
4. [Training](#Training)
5. [Evaluating the GAN](#Evaluating-the-GAN)
6. [Example Application to Earth Sciences](GANS_Tropical_Cyclones_example.ipynb)

<div style="background-color: #cce5ff; padding: 10px;">

**Import python modules**

**PLEASE NOTE MAC USERS:** Torch and Torchvision must be load BEFORE numpy and other libraries due to a Bug in PyToch > 1.1 see [GitHub Issue](https://github.com/pytorch/pytorch/issues/78490). You do not need to change anything below    

</div>


```python
# Machine Learning Library PyTorch and specific tools
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid, save_image
```


```python
# For readability: disable warnings
import warnings
warnings.filterwarnings('ignore')
```


```python
# import modules
# general file system utilites
import os
import sys
import time
```


```python
# Plotting and standard numpy
import numpy as np
from matplotlib import pyplot as plt
```

<div style="background-color: #cce5ff; padding: 10px;">

Check if GPU's available. This code will run on standard CPU's too just a lots slower.

If no GPU's are detected, then pretrained model weights will be used set by the flag

```python
train_local=False
```

**If you get a message saying no GPU available but you think there should be this means you have not managed to compile a GPU version of PyTorch and need to follow the instructions in [PyTorchCondaRecipe.md](PyTorchCondaRecipe.md)**    

</div>


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  train_local=True
  GPUs=True  
  print(torch.cuda.get_device_name(device))
  print('GPUs found Setting train_local to True')
else:
  train_local=False
  GPUs=False
  print("WARNING: No GPU available. Setting train_local to False")

```

<div style="background-color: #cce5ff; padding: 10px;">

If you want to override this behaviour and train your own network if you don't mind waiting or if you have GPUS available but don't want to wait 20 mins training your own network later you can the the next cell to:

```python
override_train_local=True
```

</div>


```python
# Set as override_train_local=False to maintain default behaviour for your hardware
override_train_local=False
```


```python
if override_train_local :
    if GPUs:
        train_local=False  
        print('WARNING SETTING TRAIN LOCAL TO FALSE, PRETRAINED MODEL WILL BE USED')
    else:
        train_local=True
        print('WARNING SETTING TRAIN LOCAL TO TRUE, WILL TRAIN MODEL OVER A LONG TIME ON CPUS')
else:
    print('Default behaviour')
```

<div style="background-color: #cce5ff; padding: 10px;">
Create data and image dir
</div>


```python
dataset_path = os.path.join(os.getcwd(), "data")
image_path = os.path.join(os.getcwd(), "images")

# Ensure both of the directories are created.
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

if not os.path.exists(image_path):
    os.mkdir(image_path)
```

# Download MNIST Dataset

<div style="background-color: #cce5ff; padding: 10px;">
With the file storage out of the way, we can define a few variables. Firstly, we need to set our batch size.
</div>


```python
batch_size = 128
```

<div style="background-color: #ccffcc; padding: 10px;">

Next let's create a transform that normalises the pixels in our image to be within the range of -1 to 1. This helps the neural network to train better than leaving the inputs unnormalised.

</div>


```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
```

<div style="background-color: #ccffcc; padding: 10px;">

We can now load our data. As mentioned above, we're using the MNIST dataset. It's a very simple dataset of 28x28 greyscale images containing handwritten digits from 0 to 9. It has been used as a benchmark since the early days of machine learning, and is simple enough that we can train a GAN on it quite quickly, despite consisting of around 60,000 samples!

</div>

<div style="background-color: #cce5ff; padding: 10px;">
PyTorch contains the MNIST dataset as a baseline, so we can just use their preformed dataset to download the data.
</div>


```python
mnist_dataset = datasets.MNIST(root=os.path.join(os.getcwd(), "data"),
                               train=True, download=True, transform=transform)
```

<div style="background-color: #cce5ff; padding: 10px;">
PyTorch uses a DataLoader wrapper around a dataset in order to iterate through the dataset. It's where we specify the number of samples per batch, as well as whether to shuffle the data. drop_last ensures that any batch that isn't at least 128 samples gets dropped.
</div>


```python
mnist_data_loader = DataLoader(mnist_dataset, batch_size=batch_size,
                               shuffle=True, drop_last=True)
```

<div style="background-color: #ccffcc; padding: 10px;">
Let's confirm how many image samples we have, as well as get a sample of images to show.
<div>


```python
print(f'Dataset size: {len(mnist_dataset)} images.')

# Get a batch from the dataloader and take 16 images from that batch to display.
batch = next(iter(mnist_data_loader))
img_batch = batch[0]
img_batch_subset = img_batch[:16]  # extract only a subset of images

# Form a grid of images and display them.
grid = make_grid(img_batch_subset, nrow=int(4), normalize=True, pad_value=1.)

# Rearrange axis from channels first to channels last for MatPlotLib.
grid = np.moveaxis(grid.numpy(), 0, 2)

# Plot our figure.
plt.figure(figsize=(6, 6))
plt.title("MNIST Samples")
plt.imshow(grid)
plt.show()
```

<div style="background-color: #ccffcc; padding: 10px;">

Now we have seen our data, we can start building our neural network!
</div>

# Building the GAN

<div style="background-color: #ccffcc; padding: 10px;">
Recall that the generator has a latent space from which we draw samples from by input random vectors. Let's define it as a vector with 100 elements and create a function to generate a batch of latent inputs.
</div>


```python
LATENT_SPACE_DIM = 100

def get_gaussian_latent_batch(batch_size, device):
    return torch.randn((batch_size, LATENT_SPACE_DIM), device=device)
```



<div style="background-color: #e6ccff; padding: 10px;">

We can now start constructing our neural networks. Remember that generative adversarial networks are actually composed of two networks: a generator and a discriminator.


Let's create a building block for constructing the networks first:
</div>


```python
def network_block(in_feat, out_feat, normalize=True, activation=None):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))

    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)

    return layers
```


<div style="background-color: #e6ccff; padding: 10px;">

This building block consists of a Linear (or fully connected) layer, a batch normalisation layer and an activation function layer (Leaky ReLU).

</div>    
<div style="background-color: #ccffcc; padding: 10px;">
Using this block, let's now build our generator, which is tasked with creating new MNIST samples:
</div>


```python
class GeneratorNet(torch.nn.Module):
    def __init__(self, img_shape=(28, 28)):
        super().__init__()
        self.generated_img_shape = img_shape
        num_neurons_per_layer = [LATENT_SPACE_DIM, 256, 512, 1024, img_shape[0] * img_shape[1]]

        # Let's define our network as a series of four of the blocks we defined above. Each time we increase the
        # number of neurons per layer and use the leaky ReLU as our activation function. For the last set of layers,
        # we skip normalisation and instead apply a hyperbolic tangent as our activation layer. This is so that the
        # output of the network is in the range of -1 to 1, with the same number of pixels as the inputs.
        self.net = nn.Sequential(
            *network_block(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            *network_block(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            *network_block(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            *network_block(num_neurons_per_layer[3], num_neurons_per_layer[4], normalize=False, activation=nn.Tanh())
        )

    def forward(self, latent_vector_batch):
        img_batch_flattened = self.net(latent_vector_batch)

        # The output of the network is a vector of 784 pixels, let's reshape it so it's a matrix of 28x28 pixels.
        return img_batch_flattened.view(img_batch_flattened.shape[0], 1, *self.generated_img_shape)
```


<div style="background-color: #e6ccff; padding: 10px;">

The above generator is built using four of the network blocks we defined above, each time increase the number of neurons per layer. The final block uses a hyperbolic tangent as its activation function, as this allows us to shape the output in the range of -1 to 1, which is the range our pixels are stored in.


Next, we'll define the discriminator in a similar manner:
</div>    


```python
class DiscriminatorNet(torch.nn.Module):
    def __init__(self, img_shape=(28, 28)):
        super().__init__()
        num_neurons_per_layer = [img_shape[0] * img_shape[1], 512, 256, 1]

        # Last layer is Sigmoid function - basically the goal of the discriminator is to output 1.
        # for real images and 0. for fake images and sigmoid is clamped between 0 and 1 so it's perfect.
        self.net = nn.Sequential(
            *network_block(num_neurons_per_layer[0], num_neurons_per_layer[1], normalize=False),
            *network_block(num_neurons_per_layer[1], num_neurons_per_layer[2], normalize=False),
            *network_block(num_neurons_per_layer[2], num_neurons_per_layer[3], normalize=False, activation=nn.Sigmoid())
        )

    def forward(self, img_batch):
        # We remove the redundant channel dimension here, since it's of size 1 (we're dealing with greyscale images
        # remember).
        img_batch_flattened = img_batch.view(img_batch.shape[0], -1)
        return self.net(img_batch_flattened)
```


<div style="background-color: #e6ccff; padding: 10px;">

We use the same block we created for the generator to keep things simple, and just reverse the process: going from an image of 28x28 to a single floating point number representing the certainty that the discriminator has of a sample being real or generated.

Now we've defined our generator and discriminator, all that remains is to create them!
    <div>


```python
# We send both networks to our training device
discriminator_net = DiscriminatorNet().to(device)
generator_net = GeneratorNet().to(device)

# Make sure they're both set to train!
discriminator_net.train()
generator_net.train()

```

# Preparing to train

<div style="background-color: #e6ccff; padding: 10px;">

Now we've created our generator and discriminator, we need to train them! As it stands, they've been initiated with random weights for all trainable parameters. This means that initially, both the generator and discriminator are going to be terrible at their respective tasks.

By training them, both will improve, and through the adversarial nature of the loss function, one network getting stronger *should* result in the other improving too. This isn't always the case however, sometimes either the generator or discriminator will become too strong and overpower the other. This is a failure case, and the model will no longer produce anything meaningful.

In the case of the generator winning, it learns a pattern that fools the discriminator while not necessarily representing the dataset at all. Often, the mode will collpase, with all latent variables converging to the same output image. If the discriminator wins, the generator loses its gradient and will begin to produce garbage outputs.

We want to avoid both cases, which makes GANs more difficult to train than most other machine learning models.

To begin our training, we first need to create an optimiser for each network. This optimiser implements an algorithm for performing backpropagation on the neural network. For this demo, we'll use the well known and effective Adam optimiser. If you want to know more about how this works, the paper it was proposed in can be found [here](https://arxiv.org/abs/1412.6980).

</div>

# Learning Rates

<div style="background-color: #e6ccff; padding: 10px;">

We set the initial learning rate for both optimisers to 0.0002. The learning rate is the single most important hyperparameter when training, and changing it can have drastic results. Try experimenting with much larger learning rates on both the generator and discriminator and see how it affects training!

</div>

<div style="background-color: #cce5ff; padding: 10px;">


**There is a link back to here at the end of this section of the tutorial.**

After you have gone through the tutorial rerun the section below with different learning rates to see the effects.

The learning rates are initally set at:
```python
    learning_rate_dis = 0.0002
    learning_rate_gen = 0.0002
```    

For example try increasing or decreasing by a factor of 10. What happens if you alter one rate but not the other?   

</div>


```python
# Learning Rate
# We set the initial learning rate for both optimisers to 0.0002.
learning_rate_dis = 0.002
learning_rate_gen = 0.00002
```


```python
discriminator_opt = Adam(discriminator_net.parameters(), lr=learning_rate_dis)
generator_opt = Adam(generator_net.parameters(), lr=learning_rate_gen)
```

<div style="background-color: #ccffcc; padding: 10px;">
In order to evaluate how the network is doing, we'll use a constant set of 16 latent inputs and monitor how the outputs change, saving a sample to the images directory.
    </div>


```python
ref_batch_size = 16
ref_noise_batch = get_gaussian_latent_batch(ref_batch_size, device)
```


<div style="background-color: #e6ccff; padding: 10px;">

We need to define the loss function that we're going to use to train our networks. We'll use binary cross entropy to measure our discriminator's predictions.

If we input real images into the discriminator we expect it to output 1 (100% certainty that the image is real).

The further away it is from 1 and the closer it is to 0 the more we should penalize it, as it is making an incorrect prediction.

BCE loss essentially becomes -log(x) when the target label is 1.

Similarly for fake images, the target label is 0 (as we want the discriminator to output 0 for fake images) and we want to penalise the generator if it starts outputing values close to 1. So we basically want to mirror the above loss function and that's just: -log(1-x).

BCE loss basically becomes -log(1-x) when it's target label is 0.

</div>    


```python
adversarial_loss = nn.BCELoss()

# Let's define our ground truth for each type of input.
real_images_gt = torch.ones((batch_size, 1), device=device)
fake_images_gt = torch.zeros((batch_size, 1), device=device)
```


<div style="background-color: #ccffcc; padding: 10px;">
Finally, let's define a few variables to help our training process.
</div>


```python
# Let's create some variables to keep track of our losses, as well as configure how often we log.
discriminator_loss_values = []
generator_loss_values = []
img_count = 0
console_log_freq = 50
debug_imagery_log_freq = 50

# This defines how many times we show the entire dataset to each network. 100 epochs means we show the networks every
# sample 100 times. Feel free to experiment with this and see if it affects training!
num_epochs = 100

# Let's also measure how much time training takes.
start_time = time.time()
```

<div style="background-color: #ccffcc; padding: 10px;">
We're now ready to begin our training loop!
</div>

# Training

<div style="background-color: #ccffcc; padding: 10px;">
Let's create our GAN training loop. We start with training the discriminator, as this helps to prevent early mode collapse.
</div>   

<div style="background-color: #ffcdcc; padding: 10px;">     

 **For 100 epochs, this training will take approximately 20-25 minutes, go grab a coffee!**

</div>    


```python
if train_local:
    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(mnist_data_loader):

            # Move the batch tensor to our training device.
            real_images = real_images.to(device)

            # Zero out the gradients in the discriminator. It's essential to do this prior to performing a forward and
            # backwards pass, else we have stale gradients from the previous iteration still present.
            discriminator_opt.zero_grad()

            # Show real images to our discriminator.
            real_discriminator_loss = adversarial_loss(discriminator_net(real_images), real_images_gt)

            # Generate fake images with the generator.
            fake_images = generator_net(get_gaussian_latent_batch(batch_size, device))

            # Show fake images to our discriminator. We call .detach() on them to detach them from the gradient graph
            # since we're not training the generator right now.
            fake_discriminator_loss = adversarial_loss(discriminator_net(fake_images.detach()), fake_images_gt)

            # Calculate the final loss by adding both terms and performing a backpropagation pass and optimiser step.
            discriminator_loss = real_discriminator_loss + fake_discriminator_loss
            discriminator_loss.backward()
            discriminator_opt.step()

            # Zero out the gradients in the generator.
            generator_opt.zero_grad()

            # Generate fake images and show them to the discriminator.
            generated_images_predictions = discriminator_net(generator_net(get_gaussian_latent_batch(batch_size, device)))

            # The generator's goal is to fool the discriminator, so our loss function should aim to make the
            # discriminator classify these fake images as real.
            generator_loss = adversarial_loss(generated_images_predictions, real_images_gt)

            # Perform a backpropagation pass and optimiser step.
            generator_loss.backward()  # this will populate .grad vars in the G net (also in D but we won't use those)
            generator_opt.step()  # perform G weights update according to optimizer's strategy

            # Log our loss values. We detach them from the graph to ensure we don't get a memory leak.
            generator_loss_values.append(generator_loss.detach().item())
            discriminator_loss_values.append(discriminator_loss.detach().item())

            # Log details to the console.
            if batch_idx % console_log_freq == 0:
                prefix = 'GAN training: time elapsed'
                print(f"{prefix} = {(time.time() - start_time):.2f} [s] | epoch={epoch + 1} | "
                      f"batch= [{batch_idx + 1}/{len(mnist_data_loader)}]")

            # Save intermediate generator images.
            if batch_idx % debug_imagery_log_freq == 0:
                with torch.no_grad():
                    log_generated_images = generator_net(ref_noise_batch)
                    log_generated_images_resized = nn.Upsample(scale_factor=2.5, mode='nearest')(log_generated_images)
                    out_path = os.path.join(image_path, f'{str(img_count).zfill(6)}.jpg')
                    save_image(log_generated_images_resized, out_path, nrow=int(np.sqrt(ref_batch_size)), normalize=True)
                    img_count += 1
    # Trained                 
    trained = True
else:
    print('Train_local set to False loading pretrained')
    try:
        generator_net = GeneratorNet()
        generator_net.load_state_dict(torch.load('pretrained/GANS_GenN_gen_LR_' + str(learning_rate_gen) +
                                                 '_dis_LR_' + str(learning_rate_dis) + '.pt'))
        discriminator_net = DiscriminatorNet()
        discriminator_net.load_state_dict(torch.load('pretrained/GANS_DisN_gen_LR_' + str(learning_rate_gen) +
                                                 '_dis_LR_' + str(learning_rate_dis) + '.pt'))
    except FileNotFoundError:
        print('pretrained/GANS_DisN_gen_LR_' + str(learning_rate_gen) + '_dis_LR_' + str(learning_rate_dis) +
              '.pkl not found,try using the following learning rates 0.002, 0.0002 or 0.00002.')

```


```python
if train_local:
    if trained:
        torch.save(GeneratorNet.state_dict, 'pretrained/GANS_GenN_gen_LR_' + str(learning_rate_gen) +
                                                 '_dis_LR_' + str(learning_rate_dis) + '.pt')
        torch.save(DiscriminatorNet.state_dict, 'pretrained/GANS_DisN_gen_LR_' + str(learning_rate_gen) +
                                                 '_dis_LR_' + str(learning_rate_dis) + '.pt')
```

<div style="background-color: #ccffcc; padding: 10px;">

Now our training is finished, it's time to evaluate how it's doing!

If you'd like, you can check out the image samples we saved during training. Remember we created a set of constant latent inputs, meaning each image has the same input and differs only because the generator's weights have updated.

The images folder contains samples produced as training has progressed. Can you see how it's improving over time?

</div>

# Evaluating the GAN

<div style="background-color: #ccffcc; padding: 10px;">

Now we've trained our model, we can use it to generate a new handwritten digit based on the MNIST dataset, but totally unique!

To start with, we'll define a few functions for prettying up and displaying the sample images.

</div>    


```python
def postprocess_generated_img(generated_img_tensor):
    assert isinstance(generated_img_tensor, torch.Tensor), f'Expected PyTorch tensor but got {type(generated_img_tensor)}.'

    # Move the tensor from GPU to CPU, convert to numpy array, extract 0th batch, move the image channel
    # from 0th to 2nd position (CHW -> HWC)
    generated_img = np.moveaxis(generated_img_tensor.to('cpu').numpy()[0], 0, 2)

    # Since MNIST images are grayscale (1-channel only) repeat 3 times to get RGB image
    generated_img = np.repeat(generated_img,  3, axis=2)

    # Normalise from range of [-1, 1], to [0, 1].
    generated_img -= np.min(generated_img)
    generated_img /= np.max(generated_img)

    return generated_img


# This function will generate a random vector pass it to the generator which will generate a new image
# which we will just post-process and return it
def generate_from_random_latent_vector(generator):
    with torch.no_grad():  # Tells PyTorch not to compute gradients which would have huge memory footprint

        # Generate a single random (latent) vector
        latent_vector = get_gaussian_latent_batch(1, next(generator.parameters()).device)

        # Post process generator output (as it's in the [-1, 1] range, remember?)
        generated_img = postprocess_generated_img(generator(latent_vector))

    return generated_img


def display_image(numpy_image):
    assert isinstance(numpy_image, np.ndarray), f'Expected numpy array got {type(numpy_image)}.'

    # Convert to uint8 format if it isn't already.
    if numpy_image.dtype != np.uint8:
        numpy_image = (numpy_image*255).astype(np.uint8)

    # Show the image.
    plt.imshow(numpy_image)
    plt.show()
```

<div style="background-color: #cce5ff; padding: 10px;">
Now let's generate a sample image from our generator! First we need to set the generator to evaluation mode, to prevent PyTorch from generating gradients.
</div>


```python
generator_net.eval()
```

<div style="background-color: #ccffcc; padding: 10px;">
Finally, let's generate our new sample!
</div>    

<div style="background-color: #cce5ff; padding: 10px;">

**You can rerun the below cell multiple times to generate new, unique samples.**

</div>


```python
generated_img = generate_from_random_latent_vector(generator_net)
display_image(generated_img)
```




<div style="background-color: #ccffcc; padding: 10px;">    

**Now is the time to go back and test the effect of latering those [Learning Rates](#Learning-Rates)**

Please feel free to experiment with the network architectures or hyperparameters. You could also try and intentionally cause the generator or discriminator to win and see what happens to the output.

For any questions about using GANs, or how they could be applied to your own research problems please contact Caitlin Howarth at sccmho@leeds.ac.uk.
</div>    

<hr>

<hr>


<div style="background-color: #e6ccff; padding: 10px;">

<h1> Earth Science Applications for GANS </h1>

Hopefully in the future we'll add an Earth Science specific tutorial to add but hopefully you can take the information you've learnt from this tutorial to underdtand these Earth Science applications.

# Met Office using GANs to predict Tropical Cyclones

The [MetOffice](https://github.com/MetOffice/) have a Git repo example [ML-TC](https://github.com/MetOffice/ML-TC), which takes open souce [Bangladesh Tropical Cyclone Data](https://doi.org/10.1038/s41597-021-00847-5) to train a Generative Adviserial Network to output synthetic tropical cyclone data. *WARNING if you chose to run these scripts the dataset download is over **100GB**. This is not suitable for running on your home laptop*     

# IBM using GANs for digitizing Earth Surface Sketches    

IMB have a short article and video using GANS to generate realistic synthetic seismic images from hand sketches. Read the article here: [Generating Sketch-Based Realistic Seismic Images with Generative Adversarial Networks, Rodrigo S. Ferreira, Julia Noce, Dario A. B. Oliveira and Emilio Vital Brazil](https://www.researchgate.net/publication/336573727_Generating_Sketch-Based_Synthetic_Seismic_Images_With_Generative_Adversarial_Networks).

Or view the video summary taken from the [IMB blog](https://www.ibm.com/blogs/research/2019/06/good-gans/) by running the cell below:

</div>    


```python
%%html
<iframe width="560" height="315" src="https://www.youtube.com/embed/n69377yKd50" title="GANs for Good: Digitizing Sketches of the Earth’s Surface with the AI Behind Deep Fakes" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

<div style="background-color: #ccffcc; padding: 10px;">

Finally, if you have a good tutorial sized Earth Science aplication using GANs that you would like to share please get in touch via the [GitHub Repo](https://github.com/cemac/LIFD_GenerativeAdversarialNetworks).     

</div>      



```python

```
