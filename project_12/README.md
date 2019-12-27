

```
import numpy as np
import time, math
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
```


<p style="color: red;">
The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>
We recommend you <a href="https://www.tensorflow.org/guide/migrate" target="_blank">upgrade</a> now 
or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:
<a href="https://colab.research.google.com/notebooks/tensorflow_version.ipynb" target="_blank">more info</a>.</p>



    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    


#####Enable Eager execution


```
tf.enable_eager_execution()
```

https://mc.ai/tutorial-1-cifar10-with-google-colabs-free-gpu%E2%80%8A-%E2%80%8A92-5/

####Initialize the weights/parameters just like numpy
Initialization function now returns a NumPy array for eager execution.



```
def init_pytorch(shape, dtype=tf.float32, partition_info=None):
  fan = np.prod(shape[:-1])
  bound = 1 / math.sqrt(fan)
  return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)
```

####Davidnet Architecture
![David Net](https://cdn-images-1.medium.com/freeze/max/1000/1*uKqdR2jn83pOhTEMLHQpJQ.png?q=20)


```
class ConvBN(tf.keras.Model):
  def __init__(self, c_out):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
    self.drop = tf.keras.layers.Dropout(0.05)

  def call(self, inputs):
    return tf.nn.relu(self.bn(self.drop(self.conv(inputs))))
```


```
class ResBlk(tf.keras.Model):
  def __init__(self, c_out, pool, res = False):
    super().__init__()
    self.conv_bn = ConvBN(c_out)
    self.pool = pool
    self.res = res
    if self.res:
      self.res1 = ConvBN(c_out)
      self.res2 = ConvBN(c_out)

  def call(self, inputs):
    h = self.pool(self.conv_bn(inputs))
    if self.res:
      h = h + self.res2(self.res1(h))
    return h
```


```
class DavidNet(tf.keras.Model):
  def __init__(self, c=64, weight=0.125):
    super().__init__()
    pool = tf.keras.layers.MaxPooling2D()
    self.init_conv_bn = ConvBN(c)
    self.blk1 = ResBlk(c*2, pool, res = True)
    self.blk2 = ResBlk(c*4, pool)
    self.blk3 = ResBlk(c*8, pool, res = True)
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
    self.weight = weight

  def call(self, x, y):
    h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    h = self.linear(h) * self.weight
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
    return loss, correct
```

####Load and preprocess the cifar10 dataset
cifar10 images are 32x32

1. Pad the train images with 4px on each side, so that the image size becomes 40x40. (mode='reflect' ~ Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.)
2. Do batch normalization


```
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
len_train, len_test = len(x_train), len(x_test)
y_train = y_train.astype('int64').reshape(len_train)
y_test = y_test.astype('int64').reshape(len_test)

train_mean = np.mean(x_train, axis=(0,1,2))
train_std = np.std(x_train, axis=(0,1,2))

normalize = lambda x: ((x - train_mean) / train_std).astype('float32') # todo: check here
pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

x_train = normalize(pad4(x_train))
x_test = normalize(x_test)
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170500096/170498071 [==============================] - 4s 0us/step



```
model = DavidNet()
```

####Hyperparameters


```
BATCH_SIZE = 512 #@param {type:"integer"}
MOMENTUM = 0.9 #@param {type:"number"}
LEARNING_RATE = 0.4 #@param {type:"number"}
WEIGHT_DECAY = 5e-4 #@param {type:"number"}
EPOCHS = 24 #@param {type:"integer"}



```


```
batches_per_epoch = len_train//BATCH_SIZE + 1
```

####Learning Schedule


```
lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
global_step = tf.train.get_or_create_global_step()
lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/BATCH_SIZE
```

####Optimizer


```
opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)
```

####Data Augmentation
1. Do a random crop so that the image size is 32x32, same as cifar10 image size
2. Randomly flip an image horizontally (left to right). y is the seed value

This would make the model robust and prevent from overfitting


```
data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)
```

####Training
The fit or fit_generator calls are not used for training.

GradientTape records the forward pass gradient computations
In the back propagation step the recorded values are used to update the trainable parameters.
Final data is printed in a formatted output. TQDM module provides a nice progress bar indication.


```

t = time.time()
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
  train_loss = test_loss = train_acc = test_acc = 0.0
  train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(data_aug).shuffle(len_train).batch(BATCH_SIZE).prefetch(1)

  tf.keras.backend.set_learning_phase(1)
  for (x, y) in tqdm(train_set):
    with tf.GradientTape() as tape:
      loss, correct = model(x, y)

    var = model.trainable_variables
    grads = tape.gradient(loss, var)
    for g, v in zip(grads, var):
      g += v * WEIGHT_DECAY * BATCH_SIZE
    opt.apply_gradients(zip(grads, var), global_step=global_step)

    train_loss += loss.numpy()
    train_acc += correct.numpy()

  tf.keras.backend.set_learning_phase(0)
  for (x, y) in test_set:
    loss, correct = model(x, y)
    test_loss += loss.numpy()
    test_acc += correct.numpy()
    
  print('epoch:', epoch+1, 'lr:', lr_schedule(epoch+1), 'train loss:', train_loss / len_train, 'train acc:', train_acc / len_train, 'val loss:', test_loss / len_test, 'val acc:', test_acc / len_test, 'time:', time.time() - t)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.random_crop is deprecated. Please use tf.image.random_crop instead.
    



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 1 lr: 0.08 train loss: 1.6123115252685547 train acc: 0.41492 val loss: 1.343320654296875 val acc: 0.5138 time: 40.58491253852844



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 2 lr: 0.16 train loss: 0.880396572265625 train acc: 0.68762 val loss: 1.1061342071533202 val acc: 0.6559 time: 66.3628556728363



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 3 lr: 0.24 train loss: 0.6634354071044922 train acc: 0.7699 val loss: 1.0162144622802733 val acc: 0.6759 time: 92.0680193901062



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 4 lr: 0.32 train loss: 0.5679113000488282 train acc: 0.80426 val loss: 1.2067989166259765 val acc: 0.6756 time: 118.02463889122009



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 5 lr: 0.4 train loss: 0.5076230065917968 train acc: 0.82546 val loss: 0.5814243072509766 val acc: 0.8065 time: 143.5640618801117



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 6 lr: 0.37894736842105264 train loss: 0.4258741258239746 train acc: 0.85192 val loss: 0.8894630096435547 val acc: 0.7424 time: 169.08073949813843



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 7 lr: 0.35789473684210527 train loss: 0.3512538984680176 train acc: 0.87688 val loss: 0.5519296661376953 val acc: 0.8336 time: 194.7063615322113



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 8 lr: 0.33684210526315794 train loss: 0.2971465603637695 train acc: 0.89774 val loss: 0.3554082954406738 val acc: 0.8763 time: 220.21709942817688



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 9 lr: 0.31578947368421056 train loss: 0.2597758149719238 train acc: 0.90936 val loss: 0.43626298828125 val acc: 0.8589 time: 245.9571406841278



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 10 lr: 0.2947368421052632 train loss: 0.22373272659301757 train acc: 0.9227 val loss: 0.40884993743896486 val acc: 0.8686 time: 271.4738552570343



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 11 lr: 0.2736842105263158 train loss: 0.20172485092163087 train acc: 0.93014 val loss: 0.335011173248291 val acc: 0.8871 time: 297.0782001018524



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 12 lr: 0.25263157894736843 train loss: 0.17893115554809572 train acc: 0.9389 val loss: 0.4146263610839844 val acc: 0.8726 time: 322.48830699920654



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 13 lr: 0.23157894736842108 train loss: 0.15778067153930664 train acc: 0.94516 val loss: 0.3380121192932129 val acc: 0.8921 time: 348.0921607017517



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 14 lr: 0.2105263157894737 train loss: 0.13803950286865235 train acc: 0.95258 val loss: 0.31241399993896485 val acc: 0.8993 time: 373.49330496788025



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 15 lr: 0.18947368421052635 train loss: 0.12452552841186523 train acc: 0.95746 val loss: 0.2920375457763672 val acc: 0.9081 time: 398.9108188152313



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 16 lr: 0.16842105263157897 train loss: 0.10761148307800293 train acc: 0.96282 val loss: 0.27241594314575196 val acc: 0.915 time: 424.2754681110382



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 17 lr: 0.1473684210526316 train loss: 0.09360897396087646 train acc: 0.96784 val loss: 0.3113281158447266 val acc: 0.9071 time: 449.4179883003235



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 18 lr: 0.12631578947368421 train loss: 0.07992435497283935 train acc: 0.97294 val loss: 0.283972705078125 val acc: 0.9145 time: 474.78596448898315



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 19 lr: 0.10526315789473689 train loss: 0.06784431285858154 train acc: 0.97764 val loss: 0.2984198028564453 val acc: 0.9154 time: 500.27870893478394



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 20 lr: 0.08421052631578951 train loss: 0.058885672836303714 train acc: 0.9814 val loss: 0.26679674377441404 val acc: 0.9243 time: 525.5448999404907



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 21 lr: 0.06315789473684214 train loss: 0.05234533462524414 train acc: 0.98338 val loss: 0.252153092956543 val acc: 0.925 time: 551.149943113327



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 22 lr: 0.04210526315789476 train loss: 0.044920551586151124 train acc: 0.98596 val loss: 0.25437313346862794 val acc: 0.9269 time: 576.7613279819489



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 23 lr: 0.02105263157894738 train loss: 0.03930884412765503 train acc: 0.98796 val loss: 0.2485149833679199 val acc: 0.9302 time: 602.324624300003



    HBox(children=(IntProgress(value=1, bar_style='info', max=1), HTML(value='')))


    
    epoch: 24 lr: 0.0 train loss: 0.03438503934860229 train acc: 0.9904 val loss: 0.2500582733154297 val acc: 0.9299 time: 627.6012361049652

