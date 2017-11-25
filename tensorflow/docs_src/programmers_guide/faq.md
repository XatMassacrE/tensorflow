# Frequently Asked Questions
# 常见问题

This document provides answers to some of the frequently asked questions about
TensorFlow. If you have a question that is not covered here, you might find an
answer on one of the TensorFlow @{$about$community resources}.
本文将会回答一些关于 TensorFlow 常见问题。
如果这里没有你想问的问题，那么建议你去
TensorFlow @{$about$community resources} 寻找答案。


[TOC]

## Features and Compatibility
## 功能和兼容性

#### Can I run distributed training on multiple computers?
#### 我可以使用多台计算机进行分布式训练吗？

Yes! TensorFlow gained
@{$distributed$support for distributed computation} in
version 0.8. TensorFlow now supports multiple devices (CPUs and GPUs) in one or
more computers.
没问题！TensorFlow 
0.8 的版本已经支持了分布式计算 @{$distributed$support for distributed computation}。
TensorFlow 现在支持在一台或者多台计算机上的多个设备（CPUs 和 GPUs）。

#### Does TensorFlow work with Python 3?
#### TensorFlow 可以在 Python 3 的环境下工作吗？

As of the 0.6.0 release timeframe (Early December 2015), we do support Python
3.3+.
早在 2015 年 12 月份，我们发行的 0.6.0 版本就已经支持了 Python 3.3+。

## Building a TensorFlow graph
## 建立一个 TensorFlow 图

See also the
@{$python/framework$API documentation on building graphs}.
同时查看
@{$python/framework$API documentation on building graphs}。

#### Why does `c = tf.matmul(a, b)` not execute the matrix multiplication immediately?
#### 为什么 `c = tf.matmul(a, b)` 不会立即执行矩阵乘法？

In the TensorFlow Python API, `a`, `b`, and `c` are
@{tf.Tensor} objects. A `Tensor` object is
a symbolic handle to the result of an operation, but does not actually hold the
values of the operation's output. Instead, TensorFlow encourages users to build
up complicated expressions (such as entire neural networks and its gradients) as
a dataflow graph. You then offload the computation of the entire dataflow graph
(or a subgraph of it) to a TensorFlow
@{tf.Session}, which is able to execute the
whole computation much more efficiently than executing the operations
one-by-one.
在 TensorFlow 的 Python API 中，`a`，`b`，和 `c` 都是 
@{tf.Tensor} 对象。一个 `Tensor` 对象是
一个操作结果的符号句柄，但是它不会真正的包含操作输出的值。
相反，TensorFlow 还鼓励用户去构建
复杂的表达式（类似于整个神经网络和它的梯度）来作为
一个数据流图。然后你卸下整个一个 TensorFlow @{tf.Session} 的数据流图（或者它的子图）
的计算，这样它就可以更有效率的执行整个计算，而一步一步的
执行操作显然是低效的。

#### How are devices named?
#### 设备是如何命名的？

The supported device names are `"/device:CPU:0"` (or `"/cpu:0"`) for the CPU
device, and `"/device:GPU:i"` (or `"/gpu:i"`) for the *i*th GPU device.
对于 CPU 来说支持的设备名字是 `"/device:CPU:0"`（或者 `"/cpu:0"`），
第 i 个 GPU 设备则会被命名为 `"/device:GPU:i"`（或者 `"/gpu:i"`）。

#### How do I place operations on a particular device?
#### 如何在一个特定的设备上执行操作？

To place a group of operations on a device, create them within a
@{tf.device$`with tf.device(name):`} context.  See
the how-to documentation on
@{$using_gpu$using GPUs with TensorFlow} for details of how
TensorFlow assigns operations to devices, and the
@{$deep_cnn$CIFAR-10 tutorial} for an example model that
uses multiple GPUs.
为了在一个设备上执行一组操作，可以在 
@{tf.device$`with tf.device(name):`} 上下文中创建它们。
查看如何去做的这个文档 @{$using_gpu$using GPUs with TensorFlow} 
可以了解到 TensorFlow 给设备分配操作的消息信息，
@{$deep_cnn$CIFAR-10 tutorial} 这篇文档
展示了使用多个 GPUs 的示例模型。


## Running a TensorFlow computation
## 运行一个 TensorFlow 运算

See also the
@{$python/client$API documentation on running graphs}.
同时查看 
@{$python/client$API documentation on running graphs} 这个文档。

#### What's the deal with feeding and placeholders?
#### 

Feeding is a mechanism in the TensorFlow Session API that allows you to
substitute different values for one or more tensors at run time. The `feed_dict`
argument to @{tf.Session.run} is a
dictionary that maps @{tf.Tensor} objects to
numpy arrays (and some other types), which will be used as the values of those
tensors in the execution of a step.
Feeding 是 TensorFlow Session API 的一个机制，它允许你
在运行时为一个或多个 tensors 替换不同的值。@{tf.Session.run} 
的参数 `feed_dict` 是一个将 @{tf.Tensor} 对象映射到 numpy 数组（或其他的类型）
的字典，这个字典将会在执行步骤的时候作为 tensors 的数值
被使用。

Often, you have certain tensors, such as inputs, that will always be fed. The
@{tf.placeholder} op allows you
to define tensors that *must* be fed, and optionally allows you to constrain
their shape as well. See the
@{$beginners$beginners' MNIST tutorial} for an
example of how placeholders and feeding can be used to provide the training data
for a neural network.
通常情况下，当你拥有确定的 tensors 时，例如输入，那么它会一直被喂养。
@{tf.placeholder} 这个操作允许你
定义**必须**被喂养的 tensors 和 

#### What is the difference between `Session.run()` and `Tensor.eval()`?
#### `Session.run()` 和 `Tensor.eval()` 的区别是什么？

If `t` is a @{tf.Tensor} object,
@{tf.Tensor.eval} is shorthand for
@{tf.Session.run} (where `sess` is the
current @{tf.get_default_session}. The
two following snippets of code are equivalent:
如果 `t` 是一个 @{tf.Tensor} 对象，
@{tf.Tensor.eval} 是 @{tf.Session.run}（
`sess` 是当前的 @{tf.get_default_session}）的
快捷方式。下面的
两段代码是等价的：

```python
# Using `Session.run()`.
sess = tf.Session()
c = tf.constant(5.0)
print(sess.run(c))

# Using `Tensor.eval()`.
c = tf.constant(5.0)
with tf.Session():
  print(c.eval())
```

In the second example, the session acts as a
[context manager](https://docs.python.org/2.7/reference/compound_stmts.html#with),
which has the effect of installing it as the default session for the lifetime of
the `with` block. The context manager approach can lead to more concise code for
simple use cases (like unit tests); if your code deals with multiple graphs and
sessions, it may be more straightforward to make explicit calls to
`Session.run()`.
在第二个例子中，sesion 表现的像一个 
[context manager](https://docs.python.org/2.7/reference/compound_stmts.html#with)，
把它作为默认的 session 来安装，会影响整个 `with` 的生命周期。

#### Do Sessions have a lifetime? What about intermediate tensors?
#### Session 有生命周期吗？中间的 tensors 呢？

Sessions can own resources, such as
@{tf.Variable},
@{tf.QueueBase}, and
@{tf.ReaderBase}; and these resources can use
a significant amount of memory. These resources (and the associated memory) are
released when the session is closed, by calling
@{tf.Session.close}.
Session 可以拥有一些资源，例如 
@{tf.Variable}，
@{tf.QueueBase} 和 
@{tf.ReaderBase}。这些资源可以使用
一个有效数量的内存。这些资源（以及相关的内存）都是通过
调用 @{tf.Session.close} 在 
session 被关闭的时候释放的，

The intermediate tensors that are created as part of a call to
@{$python/client$`Session.run()`} will be freed at or before the
end of the call.
作为调用 @{$python/client$`Session.run()`} 的一部分
被创建的中间的 tensors 将会在调用或者调用之前
被释放掉。

#### Does the runtime parallelize parts of graph execution?
#### Graph 执行在运行时是并行的吗？

The TensorFlow runtime parallelizes graph execution across many different
dimensions:
TensorFlow 运行时的并行图执行穿插着很多不同的维度：

* The individual ops have parallel implementations, using multiple cores in a
  CPU, or multiple threads in a GPU.
* 单独的操作有并发的实现，它会使用的 CPU 的多个核心
  或者一个 GPU 的多个线程。
* Independent nodes in a TensorFlow graph can run in parallel on multiple
  devices, which makes it possible to speed up
  @{$deep_cnn$CIFAR-10 training using multiple GPUs}.
* TensorFlow graph 中的独立节点可以在多个设备上
  同时运行，这样才能加速 
  @{$deep_cnn$CIFAR-10 training using multiple GPUs}。
* The Session API allows multiple concurrent steps (i.e. calls to
  @{tf.Session.run} in parallel. This
  enables the runtime to get higher throughput, if a single step does not use
  all of the resources in your computer.
* Session API 允许多个并发的步骤（例如，
  并行的调用 @{tf.Session.run}。这样可以在运行时
  获得更大的吞吐量，尤其是当一个单独步骤
  不会使用到你计算机的全部资源时。

#### Which client languages are supported in TensorFlow?
#### TensorFlow 支持哪些客户端语言？

TensorFlow is designed to support multiple client languages.
Currently, the best-supported client language is [Python](../api_docs/python/index.md). Experimental interfaces for
executing and constructing graphs are also available for
[C++](../api_docs/cc/index.md), [Java](../api_docs/java/reference/org/tensorflow/package-summary.html) and [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go).
TensorFlow 就是为支持多客户端语言设计的。
现在，我们支持的最好的客户端语言是 [Python](../api_docs/python/index.md)。执行和构建 graphs 的一些实验性的接口
对于 [C++](../api_docs/cc/index.md)，[Java](../api_docs/java/reference/org/tensorflow/package-summary.html) 和 [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go) 也是可用的。

TensorFlow also has a
[C-based client API](https://www.tensorflow.org/code/tensorflow/c/c_api.h)
to help build support for more client languages.  We invite contributions of new
language bindings.
同时 TensorFlow 还有一个 
[C-based client API](https://www.tensorflow.org/code/tensorflow/c/c_api.h)，
它可以帮助我们创建和支持更多的客户端语言。我们也邀请了一些贡献者
一起完成对新语言的支持。

Bindings for various other languages (such as [C#](https://github.com/migueldeicaza/TensorFlowSharp), [Julia](https://github.com/malmaud/TensorFlow.jl), [Ruby](https://github.com/somaticio/tensorflow.rb) and [Scala](https://github.com/eaplatanios/tensorflow_scala)) created and supported by the opensource community build on top of the C API supported by the TensorFlow maintainers.
支持其他各种语言（例如 [C#](https://github.com/migueldeicaza/TensorFlowSharp)，[Julia](https://github.com/malmaud/TensorFlow.jl)，
[Ruby](https://github.com/somaticio/tensorflow.rb) 和 [Scala](https://github.com/eaplatanios/tensorflow_scala)）的工作，将会由开源社区来创建和支持，
这些支持都是基于 TensorFlow 维护者的提供的 C API 来构建的。

#### Does TensorFlow make use of all the devices (GPUs and CPUs) available on my machine?
#### TensorFlow 可以使用我机器上的所有设备（GPUs 和 CPUs）吗？

TensorFlow supports multiple GPUs and CPUs. See the how-to documentation on
@{$using_gpu$using GPUs with TensorFlow} for details of how
TensorFlow assigns operations to devices, and the
@{$deep_cnn$CIFAR-10 tutorial} for an example model that
uses multiple GPUs.
TensorFlow 支持多个 GPUs 和 CPUs。查看 how-to 文档（
@{$using_gpu$using GPUs with TensorFlow}）可以获得
关于 TensorFlow 是如何给设备分配操作的详细信息。并且 
@{$deep_cnn$CIFAR-10 tutorial} 还提供了一个
使用多个 GPUs 的示例模型。

Note that TensorFlow only uses GPU devices with a compute capability greater
than 3.5.
注意：TensorFlow 仅仅使用计算性能大于 3.5 的 GPU 设备。

#### Why does `Session.run()` hang when using a reader or a queue?
#### 为什么使用一个读取器或者队列的时候 `Session.run()` 会挂起？

The @{tf.ReaderBase} and
@{tf.QueueBase} classes provide special operations that
can *block* until input (or free space in a bounded queue) becomes
available. These operations allow you to build sophisticated
@{$reading_data$input pipelines}, at the cost of making the
TensorFlow computation somewhat more complicated. See the how-to documentation
for
@{$reading_data#creating-threads-to-prefetch-using-queuerunner-objects$using
`QueueRunner` objects to drive queues and readers}
for more information on how to use them.
@{tf.ReaderBase} 和 @{tf.QueueBase} 类提供了一些可以 
*阻塞* 进程直到输入可用（或者有界队列释放出空间）
的特殊操作。这些操作可以让你建立复杂的 
@{$reading_data$input pipelines}，当然了，这样做的代价
就是 TensorFlow 的计算也会变的复杂。看看 how-to 文档吧：
@{$reading_data#creating-threads-to-prefetch-using-queuerunner-objects$using 
`QueueRunner` objects to drive queues and readers}，
它会告诉你如何使用它们。

## Variables
## 变量

See also the how-to documentation on @{$variables$variables} and
@{$python/state_ops$the API documentation for variables}.
同时查看关于 @{$variables$variables} 和 
@{$python/state_ops$the API documentation for variables} 的 how-to 文档。

#### What is the lifetime of a variable?
#### 什么是一个变量的生命周期？

A variable is created when you first run the
@{tf.Variable.initializer}
operation for that variable in a session. It is destroyed when that
@{tf.Session.close}.
当你第一次在 session 对变量执行 
@{tf.Variable.initializer} 这个操作时，
这个变量就会被创建。当执行 
@{tf.Session.close} 这个操作时，它就会被销毁。

#### How do variables behave when they are concurrently accessed?
#### 当这些变量被并发的调用时，它们的表现如何？

Variables allow concurrent read and write operations. The value read from a
variable may change if it is concurrently updated. By default, concurrent
assignment operations to a variable are allowed to run with no mutual exclusion.
To acquire a lock when assigning to a variable, pass `use_locking=True` to
@{tf.Variable.assign}.
变量允许并发的执行读和写操作。但是在被并发更新的时候
从一个变量中读取的数值可能会改变。默认情况下，在没有共同排斥的前提下，对一个变量
并发的分配操作是没问题的。
通过给 @{tf.Variable.assign} 传递 @{tf.Variable.assign} 这样一个参数，
可以在分配变量时获得一个锁。

## Tensor shapes

See also the
@{tf.TensorShape}.
同时查看 
@{tf.TensorShape}。

#### How can I determine the shape of a tensor in Python?
#### 在 Python 中我如何决定一个 tensor 的 shape？

In TensorFlow, a tensor has both a static (inferred) shape and a dynamic (true)
shape. The static shape can be read using the
@{tf.Tensor.get_shape}
method: this shape is inferred from the operations that were used to create the
tensor, and may be
@{tf.TensorShape$partially complete}. If the static
shape is not fully defined, the dynamic shape of a `Tensor` `t` can be
determined by evaluating @{tf.shape$`tf.shape(t)`}.
在 TensorFlow 中，一个 tensor 同时拥有一个静态的（推测出的）的 shape 和 一个
动态的（真实的）的 shape。这个静态的 shape 可以使用 
@{tf.Tensor.get_shape} 
方法读取到：这个 shape 是通过我们过去创建 tensor 的操作推测出来的，
也有可能是通过 
@{tf.TensorShape$partially complete}。如果静态的 shape 
并没有被完全的定义，那么一个 `Tensor` 的动态 shape `t` 可以通过
估计 @{tf.shape$`tf.shape(t)`} 来决定。

#### What is the difference between `x.set_shape()` and `x = tf.reshape(x)`?
#### `x.set_shape()` 和 `x = tf.reshape(x)` 的区别是什么？

The @{tf.Tensor.set_shape} method updates
the static shape of a `Tensor` object, and it is typically used to provide
additional shape information when this cannot be inferred directly. It does not
change the dynamic shape of the tensor.
@{tf.Tensor.set_shape} 方法会更新一个 `Tensor` 对象的静态 shape，
无法通过直接的推测获取 shape 信息时，
这个方法就是最典型的获得额外的 shape 信息的方法了。当然了，它并不会
改变这个 tensor 的动态 shape。

The @{tf.reshape} operation creates
a new tensor with a different dynamic shape.
@{tf.reshape} 这个操作会使用一个不同的动态 shape 来
创建一个新的 tensor。

#### How do I build a graph that works with variable batch sizes?
#### 如何才能创建一个

It is often useful to build a graph that works with variable batch sizes, for
example so that the same code can be used for (mini-)batch training, and
single-instance inference. The resulting graph can be
@{tf.Graph.as_graph_def$saved as a protocol buffer}
and
@{tf.import_graph_def$imported into another program}.
建立一个，
这样，同样的代码即可以使用在（mini-）批训练，又可以使用在
单实例接口上。作为结果的 graph 可以是 
@{tf.Graph.as_graph_def$saved as a protocol buffer} 
和 
@{tf.import_graph_def$imported into another program}。

When building a variable-size graph, the most important thing to remember is not
to encode the batch size as a Python constant, but instead to use a symbolic
`Tensor` to represent it. The following tips may be useful:
当创建一个变化大小的 graph 时，最重要的事情就是不要把
批处理的大小编码成一个 Python 的常量，而是要使用一个
符号 `Tensor` 去代表它。下面的这些 tips 或许会对你有帮助：

* Use [`batch_size = tf.shape(input)[0]`](../api_docs/python/array_ops.md#shape)
  to extract the batch dimension from a `Tensor` called `input`, and store it in
  a `Tensor` called `batch_size`.
* 使用 [`batch_size = tf.shape(input)[0]`](../api_docs/python/array_ops.md#shape) 
  来从一个 `Tensor` 中解压出叫做 `input` 的多维数据，并且把它
  储存在一个叫 `batch_size` 的 `Tensor` 中。

* Use @{tf.reduce_mean} instead
  of `tf.reduce_sum(...) / batch_size`.
* 使用 @{tf.reduce_mean} 
  来代替 `tf.reduce_sum(...) / batch_size`。


## TensorBoard

#### How can I visualize a TensorFlow graph?
#### 如何可视化一个 TensorFlow graph ？

See the @{$graph_viz$graph visualization tutorial}.
查看 @{$graph_viz$graph visualization tutorial}。

#### What is the simplest way to send data to TensorBoard?
#### 向 TensorBoard 发送数据的最简单方法是什么？

Add summary ops to your TensorFlow graph, and write
these summaries to a log directory.  Then, start TensorBoard using
给你的 TensorFlow graph 添加摘要的操作，并且
把这些摘要写在一个日志文件中。然后使用下面的命令来开始 TensorBoard：

    python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory

For more details, see the
@{$summaries_and_tensorboard$Summaries and TensorBoard tutorial}.
更多详细的信息，请查看 
@{$summaries_and_tensorboard$Summaries and TensorBoard tutorial}。

#### Every time I launch TensorBoard, I get a network security popup!
#### 每次我登录到 TensorBoard 时，就会得到一个网络安全的弹出框！

You can change TensorBoard to serve on localhost rather than '0.0.0.0' by
the flag --host=localhost. This should quiet any security warnings.
你可以使用 --host=localhost 这个参数将 TensorBoard 运行在 localhost，
而不是 '0.0.0.0'。这样就不会有安全警告了。

## Extending TensorFlow
## 拓展 TensorFlow

See the how-to documentation for
@{$adding_an_op$adding a new operation to TensorFlow}.
查看 how-to 文档 
@{$adding_an_op$adding a new operation to TensorFlow}。

#### My data is in a custom format. How do I read it using TensorFlow?
#### 我的数据是自定义的格式，怎样做才能让 TensorFlow 正确的读取它们呢？

There are three main options for dealing with data in a custom format.
对于自定义格式的数据，我们有一些主要的选项来处理。

The easiest option is to write parsing code in Python that transforms the data
into a numpy array. Then use @{tf.data.Dataset.from_tensor_slices} to
create an input pipeline from the in-memory data.
最简单的选项就是使用 Pyhton 编写解析的代码，将数据转换成 numpy 的 array。
然后使用 @{tf.data.Dataset.from_tensor_slices} 来
从内存数据中创建出一个输入的 pipeline。

If your data doesn't fit in memory, try doing the parsing in the Dataset
pipeline. Start with an appropriate file reader, like
@{tf.data.TextLineDataset}. Then convert the dataset by mapping
@{tf.data.Dataset.map$mapping} appropriate operations over it.
Prefer predefined TensorFlow operations such as @{tf.decode_raw},
@{tf.decode_csv}, @{tf.parse_example}, or @{tf.image.decode_png}.
如果你的数据不适合放在内存中，那么可以试试在数据集
的 pipeline 中解析。使用一个合适的文件读取器，
像 @{tf.data.TextLineDataset}。然后通过映射 
@{tf.data.Dataset.map$mapping} 适当的操作来转换数据集。
最好是预定义 TensorFlow 的一些操作，像 @{tf.decode_raw}，
@{tf.decode_csv}，@{tf.parse_example} 或者 @{tf.image.decode_png}。

If your data is not easily parsable with the built-in TensorFlow operations,
consider converting it, offline, to a format that is easily parsable, such
as ${tf.python_io.TFRecordWriter$`TFRecord`} format.
如果你的数据不太好用 TensorFlow 内建的一些操作来解析，
那么考虑下转换它吧，在离线模式下转换成一种容易被解析的格式，
比如 ${tf.python_io.TFRecordWriter$`TFRecord`} 格式。

The more efficient method to customize the parsing behavior is to
@{$adding_an_op$add a new op written in C++} that parses your
data format. The @{$new_data_formats$guide to handling new data formats} has
more information about the steps for doing this.
自定义解析行为的更有效的方法是 @{$adding_an_op$add a new op written in C++}，
这个新添加的操作是可以解析你的数据格式的。
@{$new_data_formats$guide to handling new data formats} 有
关于如何操作的更详细的步骤。


## Miscellaneous
## 其他

#### What is TensorFlow's coding style convention?
#### TensorFlow 的代码风格是什么样的？

The TensorFlow Python API adheres to the
[PEP8](https://www.python.org/dev/peps/pep-0008/) conventions.<sup>*</sup> In
particular, we use `CamelCase` names for classes, and `snake_case` names for
functions, methods, and properties. We also adhere to the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html).
TensorFlow Python API 的代码风格是坚持 
[PEP8](https://www.python.org/dev/peps/pep-0008/) 的约定的。<sup>*</sup>
要注意的是，我们使用 `CamelCase` 来对类进行命名，使用 `snake_case` 来对
函数，方法以及属性进行命名。同时我们也坚持 
[Google Python style guide](https://google.github.io/styleguide/pyguide.html)。

The TensorFlow C++ code base adheres to the
[Google C++ style guide](http://google.github.io/styleguide/cppguide.html).
TensorFlow C++ 代码风格坚持 
[Google C++ style guide](http://google.github.io/styleguide/cppguide.html)。

(<sup>*</sup> With one exception: we use 2-space indentation instead of 4-space
indentation.)
(<sup>*</sup>有一个例外是：我们使用 2 个空格进行缩进，而不是 4 个。)



