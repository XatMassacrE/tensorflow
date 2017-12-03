# Vector Representations of Words
# 用向量代表单词

In this tutorial we look at the word2vec model by
[Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
This model is used for learning vector representations of words, called "word
embeddings".
本文让我们来通过 
[Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 
来看看 word2vec 这个模型，它被用来学习叫做『单词嵌入』的功能，
这个功能可以使用向量来代表单词。

## Highlights
## 集锦

This tutorial is meant to highlight the interesting, substantive parts of
building a word2vec model in TensorFlow.
本文收录了用 TensorFlow 构建一个 word2vec 模型的部分有趣真实的细节。

* We start by giving the motivation for why we would want to
represent words as vectors.
* 首先我们需要知道为什么我们想要
  使用向量来代表单词。
* We look at the intuition behind the model and how it is trained
(with a splash of math for good measure).
* 其次我们需要看到模型背后的直觉以及它是如何被训练的
  （使用数学的奇技淫巧来进行测量）。
* We also show a simple implementation of the model in TensorFlow.
* 同时我们也会使用 TensorFlow 实现一个简单的模型。
* Finally, we look at ways to make the naive version scale better.
* 最后，我们会看到让本地版本的拓展的更好的各种方法。

We walk through the code later during the tutorial, but if you'd prefer to dive
straight in, feel free to look at the minimalistic implementation in
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
This basic example contains the code needed to download some data, train on it a
bit and visualize the result. Once you get comfortable with reading and running
the basic version, you can graduate to
[models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py)
which is a more serious implementation that showcases some more advanced
TensorFlow principles about how to efficiently use threads to move data into a
text model, how to checkpoint during training, etc.
本文只会粗略的过一遍代码，如果你想深入了解的话，
可以去 [tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) 
看到一个极简的实现。
这个基础的示例包含需要下载的数据的代码，稍微训练一下这些数据
然后把结果可视化出来。一旦你可以轻松的阅读并运行这个基础的版本，
你就可以去看 
[models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py) 了，
这是一个更严谨的实现，它展示了一些更高级的 TensorFlow 原则，
例如如何高效的使用线程
来将数据移动到一个文本模型中，以及如何在训练中设置检查点等。

But first, let's look at why we would want to learn word embeddings in the first
place. Feel free to skip this section if you're an Embedding Pro and you'd just
like to get your hands dirty with the details.
首先，让我们来看一看为什么我们想要学习词向量。
如果你是一个词嵌入专家的话，可以直接跳过本节，不然
看到这些细节你会觉得很无聊。

## Motivation: Why Learn Word Embeddings?
## 动机：为什么要学习词向量？

Image and audio processing systems work with rich, high-dimensional datasets
encoded as vectors of the individual raw pixel-intensities for image data, or
e.g. power spectral density coefficients for audio data. For tasks like object
or speech recognition we know that all the information required to successfully
perform the task is encoded in the data (because humans can perform these tasks
from the raw data).  However, natural language processing systems traditionally
treat words as discrete atomic symbols, and therefore 'cat' may be represented
as  `Id537` and 'dog' as `Id143`.  These encodings are arbitrary, and provide
no useful information to the system regarding the relationships that may exist
between the individual symbols. This means that the model can leverage
very little of what it has learned about 'cats' when it is processing data about
'dogs' (such that they are both animals, four-legged, pets, etc.). Representing
words as unique, discrete ids furthermore leads to data sparsity, and usually
means that we may need more data in order to successfully train statistical
models.  Using vector representations can overcome some of these obstacles.
图像和音频处理系统在处理丰富，高维度的数据集时会将
图像的单个像素信号量或
音频的功率谱密度系数编码成向量。对于图像识别或者
语音识别这样的任务来说，我们知道要成功的识别出正确结果的所有信息
都在数据里（因为人类可以从这些原始数据中得到答案）。
然而，传统的自然语言处理系统将
单词看做确定的原子符号，因此 `Id537` 会代表 'cat'，
`Id143` 会代表 'dog'。这些编码都是任意的，
并且它们对于单个字符之间的关系系统
毫无帮助。这意味着当模型处理关于 'dogs' 数据的时候，
基本上用不到它从处理 'cats' 上学到的知识
（然而它们有很多共同点，都是动物，四条腿，宠物等）。
将单词表示成唯一离散的 id，这样会使数据变的稀疏，
通常情况下这意味着为了成功的训练统计模型，我们需要更多的数据。
而使用向量代表单词就可以克服这些困难。


<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/audio-image-text.png" alt>
</div>

[Vector space models](https://en.wikipedia.org/wiki/Vector_space_model) (VSMs)
represent (embed) words in a continuous vector space where semantically
similar words are mapped to nearby points ('are embedded nearby each other').
VSMs have a long, rich history in NLP, but all methods depend in some way or
another on the
[Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis),
which states that words that appear in the same contexts share
semantic meaning. The different approaches that leverage this principle can be
divided into two categories: *count-based methods* (e.g.
[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)),
and *predictive methods* (e.g.
[neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models)).

[Vector space models](https://en.wikipedia.org/wiki/Vector_space_model) (VSMs)
是在一个连续的向量空间中代表单词，
并且把语义上相似的单词映射到它附近的点
（就是这些单词彼此都靠的很近）。
VSMs 在 NLP 领域中历史悠久，但是它依赖的所有方法在某种程度上
都是分享语义上的意思，或者声明相同上下文中出现的单词 
[Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis)。
而另一个不同的利用这个原则的方法是
深入下面两个分类：*计数方法*（例如：
[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis))，
和 *预测方法*（例如：
[neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models))。

This distinction is elaborated in much more detail by
[Baroni et al.](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf),
but in a nutshell: Count-based methods compute the statistics of
how often some word co-occurs with its neighbor words in a large text corpus,
and then map these count-statistics down to a small, dense vector for each word.
Predictive models directly try to predict a word from its neighbors in terms of
learned small, dense *embedding vectors* (considered parameters of the
model).
[Baroni et al.](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf) 
更详细的阐述了这种区别，
但是一言以蔽之就是：计数方法会计算在一个大文本集中一些单词以及它们类似单词一起出现的频率，
然后将每一个单词的
这些统计数据降维到一个小而密的向量。
预测模型则是根据学习到的小而密的*词向量*（模型深思熟虑的参数）来
直接从它的邻近单词中预测出单词。

Word2vec is a particularly computationally-efficient predictive model for
learning word embeddings from raw text. It comes in two flavors, the Continuous
Bag-of-Words model (CBOW) and the Skip-Gram model (Section 3.1 and 3.2 in [Mikolov et al.](https://arxiv.org/pdf/1301.3781.pdf)). Algorithmically, these
models are similar, except that CBOW predicts target words (e.g. 'mat') from
source context words ('the cat sits on the'), while the skip-gram does the
inverse and predicts source context-words from the target words. This inversion
might seem like an arbitrary choice, but statistically it has the effect that
CBOW smoothes over a lot of the distributional information (by treating an
entire context as one observation). For the most part, this turns out to be a
useful thing for smaller datasets. However, skip-gram treats each context-target
pair as a new observation, and this tends to do better when we have larger
datasets. We will focus on the skip-gram model in the rest of this tutorial.
对于从源数据中学习词向量来说，Word2vec 是一个计算效率很高的预测模型。
这其中包含两层含义，
Continuous Bag-of-Words（CBOW）和 Skip-Gram 模型（[Mikolov et al.](https://arxiv.org/pdf/1301.3781.pdf) 中的 3.1 和 3.2 章节）。从算法角度来说，
这些模型是类似的，CBOW 是从源上下文数据（'the cat sits on the'）中预测出目标单词（比如 'mat'），
skip-gram 则相反，它是从目标单词中预测出源上下文数据。这种反向操作可能
看起来很随意，但是从统计的角度讲，它是有效的，因为 CBOW 可以消除
很多分散的信息（通过将整个上下文看做一个观察点）。在大多数情况下，这种操作
对于小一点的数据集也是适用的。然而，skip-gram 将每一个上下文目标对
都看做一个新的观察点，这样对于更大的数据集是更有利的。
在下面的篇幅中，我们将主要来看 skip-gram 这个模型。


## Scaling up with Noise-Contrastive Training
## 放大对比噪音训练

Neural probabilistic language models are traditionally trained using the
[maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood) (ML)
principle  to maximize the probability of the next word \\(w_t\\) (for "target")
given the previous words \\(h\\) (for "history") in terms of a
[*softmax* function](https://en.wikipedia.org/wiki/Softmax_function),
神经概率语言模型传统的使用 
[maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood) (ML) 
定理来最大化下一个单词的的概率，根据 
[*softmax* function](https://en.wikipedia.org/wiki/Softmax_function) 这个函数
使用之前的单词 \\(h\\) (代表 "history") 来推出
下一个单词 \\(w_t\\) (代表 "target")。

$$
\begin{align}
P(w_t | h) &= \text{softmax}(\text{score}(w_t, h)) \\
           &= \frac{\exp \{ \text{score}(w_t, h) \} }
             {\sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} }
\end{align}
$$

where \\(\text{score}(w_t, h)\\) computes the compatibility of word \\(w_t\\)
with the context \\(h\\) (a dot product is commonly used). We train this model
by maximizing its [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function)
on the training set, i.e. by maximizing
在这里 \\(\text{score}(w_t, h)\\) 计算出了在上下文为 \\(h\\)（通常使用点积）
的情况下单词 \\(w_t\\) 的兼容性。我们通过在训练集上
最大化它的 [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function)
来训练这个模型，比如通过最大化

$$
\begin{align}
 J_\text{ML} &= \log P(w_t | h) \\
  &= \text{score}(w_t, h) -
     \log \left( \sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} \right).
\end{align}
$$

This yields a properly normalized probabilistic model for language modeling.
However this is very expensive, because we need to compute and normalize each
probability using the score for all other \\(V\\) words \\(w'\\) in the current
context \\(h\\), *at every training step*.
这样会为语言建模产生一个正确标准化的概率模型。
然而这样做的代价也是昂贵的，因为我们需要使用
当前上下文 \\(h\\) 中所有其他的 \\(V\\) 单词 \\(w'\\) 的得分来
对*每一步的训练*都计算和标准化每一个概率。

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-nplm.png" alt>
</div>

On the other hand, for feature learning in word2vec we do not need a full
probabilistic model. The CBOW and skip-gram models are instead trained using a
binary classification objective ([logistic regression](https://en.wikipedia.org/wiki/Logistic_regression))
to discriminate the real target words \\(w_t\\) from \\(k\\) imaginary (noise) words \\(\tilde w\\), in the
same context. We illustrate this below for a CBOW model. For skip-gram the
direction is simply inverted.
另一方面，对于 word2vec 的特征学习来说，我们不需要一个完整的概率模型。
在同样的上下文中，CBOW 和 skip-gram 模型采用了二元
分类 ([logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)) 来
从 \\(k\\) 以及虚假的（噪音）单词 \\(\tilde w\\) 中区别出真正的目标单词 \\(w_t\\)。
我们会在下面使用 CBOW 模型来示范。对于 skip-gram 模型来说，
仅仅改变成相反的方向就可以了。

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/nce-nplm.png" alt>
</div>

Mathematically, the objective (for each example) is to maximize
从数学的角度讲，每一个例子的目标都是去最大化

$$J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) +
  k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right]$$

where \\(Q_\theta(D=1 | w, h)\\) is the binary logistic regression probability
under the model of seeing the word \\(w\\) in the context \\(h\\) in the dataset
\\(D\\), calculated in terms of the learned embedding vectors \\(\theta\\). In
practice we approximate the expectation by drawing \\(k\\) contrastive words
from the noise distribution (i.e. we compute a
[Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration)).
这里，\\(Q_\theta(D=1 | w, h)\\) 表示在数据集 \\(D\\) 的上下文 \\(h\\) 中
有单词 \\(w\\) 的模型的二元逻辑回归概率，
它是根据学习之后的词向量 \\(\theta\\) 来计算的。
实际上我们会通过在噪音分布中画出对比的单词
来近似的得到期望（比如我们可以计算一个
[Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration))。

This objective is maximized when the model assigns high probabilities
to the real words, and low probabilities to noise words. Technically, this is
called
[Negative Sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf),
and there is good mathematical motivation for using this loss function:
The updates it proposes approximate the updates of the softmax function in the
limit. But computationally it is especially appealing because computing the
loss function now scales only with the number of *noise words* that we
select (\\(k\\)), and not *all words* in the vocabulary (\\(V\\)). This makes it
much faster to train. We will actually make use of the very similar
[noise-contrastive estimation (NCE)](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)
loss, for which TensorFlow has a handy helper function `tf.nn.nce_loss()`.
这样做的目的就是最大化这个期望，模型会给真正的单词分配一个高的概率，
给噪音单词分配一个低的概率。从学术角度说，
它被称作 
[Negative Sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)，
并且这里对使用这个损失函数还有良好的数学激励：


Let's get an intuitive feel for how this would work in practice!
让我们从直觉上感受一下在实践中是如何工作的！

## The Skip-gram Model
## Skip-gram 模型

As an example, let's consider the dataset
举个例子，让我们考虑一下下面的数据集

`the quick brown fox jumped over the lazy dog`

We first form a dataset of words and the contexts in which they appear. We
could define 'context' in any way that makes sense, and in fact people have
looked at syntactic contexts (i.e. the syntactic dependents of the current
target word, see e.g.
[Levy et al.](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)),
words-to-the-left of the target, words-to-the-right of the target, etc. For now,
let's stick to the vanilla definition and define 'context' as the window
of words to the left and to the right of a target word. Using a window
size of 1, we then have the dataset
首先让我们构建一个单词数据集以及这些单词所在的上下文环境。
我们可以以任何合理的方式定义这个 'context'，而且实际上人们
查看语法的上下文（比如依赖当前目标单词的语法，可以
查看 [Levy et al.](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)），
以及目标左边的单词，目标右边的单词等。现在坚持使用单纯的定义并
把目标左边的单词和目标右边的单词作为窗口定义为 'context'。
使用一个大小为 1 的窗口，我们得到下面的数据集：

`([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...`

of `(context, target)` pairs. Recall that skip-gram inverts contexts and
targets, and tries to predict each context word from its target word, so the
task becomes to predict 'the' and 'brown' from 'quick', 'quick' and 'fox' from
'brown', etc. Therefore our dataset becomes
即 `(context, target)` 这样的数据对。回想一下，skip-gram 会反转上下文
和目标，并且从目标单词中试着预测每一个上下文的单词，所以我们的任务
变成了从 'quick' 预测 'the' 和 'brown'，从 'brown' 预测 'quick' 和 'fox' 等。
因此我们的数据集就变成了下面这样：

`(quick, the), (quick, brown), (brown, quick), (brown, fox), ...`

of `(input, output)` pairs.  The objective function is defined over the entire
dataset, but we typically optimize this with
[stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
(SGD) using one example at a time (or a 'minibatch' of `batch_size` examples,
where typically `16 <= batch_size <= 512`). So let's look at one step of
this process.
即 `(input, output)` 这样的数据对。这个目标函数是基于整个数据集定义的，
但是我们可以使用 
[stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)(SGD) 来
优化这个目标函数，每次使用一个样本（或者在 `16 <= batch_size <= 512` 这个
条件下，使用 `batch_size` 中的一个 'minibatch'）。所以让我们
看看这个过程的一个步骤。

Let's imagine at training step \\(t\\) we observe the first training case above,
where the goal is to predict `the` from `quick`. We select `num_noise` number
of noisy (contrastive) examples by drawing from some noise distribution,
typically the unigram distribution, \\(P(w)\\). For simplicity let's say
`num_noise=1` and we select `sheep` as a noisy example. Next we compute the
loss for this pair of observed and noisy examples, i.e. the objective at time
step \\(t\\) becomes
让我们想象一下在训练步骤 \\(t\\) 我们观察上面训练的第一个情况，
就是目标是从 `quick` 预测出 `the` 的训练。我们选择了 `num_noise` 个
噪音（对比的）样本，这些样本是通过均匀分布 \\(P(w)\\) 的噪音中
画出来的。简单来说就是我们设置 `num_noise=1`，并且选择 `sheep` 作为
噪音样本。下一步我们就可以计算
这个观察样本和噪音样本对的损失函数了。例如，
此时我们的目标就是：

$$J^{(t)}_\text{NEG} = \log Q_\theta(D=1 | \text{the, quick}) +
  \log(Q_\theta(D=0 | \text{sheep, quick}))$$

The goal is to make an update to the embedding parameters \\(\theta\\) to improve
(in this case, maximize) this objective function.  We do this by deriving the
gradient of the loss with respect to the embedding parameters \\(\theta\\), i.e.
\\(\frac{\partial}{\partial \theta} J_\text{NEG}\\) (luckily TensorFlow provides
easy helper functions for doing this!). We then perform an update to the
embeddings by taking a small step in the direction of the gradient. When this
process is repeated over the entire training set, this has the effect of
'moving' the embedding vectors around for each word until the model is
successful at discriminating real words from noise words.
这个目标就是通过更新嵌入的参数 \\(\theta\\) 来提升（这里就是最大化）
这个结果。为此，我们需要在考虑到嵌入参数 \\(\theta\\) 的情况下得到梯度的损失值，
比如 \\(\frac{\partial}{\partial \theta} J_\text{NEG}\\)（幸运的是，TensorFlow 提供了
非常简单的辅助函数来帮助我们完成计算！）。然后我们再执行一次更新，
向梯度方向移动一小步。当整个训练集都完成这个过程之后，直到模型可以
成功的从噪音单词中分辨出真实单词时，这个模型才会对
每一个单词周围『绕来绕去』的词向量有影响。

We can visualize the learned vectors by projecting them down to 2 dimensions
using for instance something like the
[t-SNE dimensionality reduction technique](https://lvdmaaten.github.io/tsne/).
When we inspect these visualizations it becomes apparent that the vectors
capture some general, and in fact quite useful, semantic information about
words and their relationships to one another. It was very interesting when we
first discovered that certain directions in the induced vector space specialize
towards certain semantic relationships, e.g. *male-female*, *verb tense* and
even *country-capital* relationships between words, as illustrated in the figure
below (see also for example
[Mikolov et al., 2013](https://www.aclweb.org/anthology/N13-1090)).
我们可以先使用一些像 
[t-SNE dimensionality reduction technique](https://lvdmaaten.github.io/tsne/) 的实例
来将学习之后的向量降到 2 维，然后我们就可以对它们进行可视化了。
当我观察这些可视化的数据时，这些向量捕获的通用的，并且非常有效的关于
单词和与单词之间关系的语义信息就变的显而易见了。
当我们第一次发现在诱发的向量中某个特定方向会指向一个特定的语义
关系时，确实非常有趣，
比如  *male-female*， *verb tense*， 甚至 *country-capital* 这些单词之间的关系，
就像下图展示的那样
（也可以去查看 
[Mikolov et al., 2013](https://www.aclweb.org/anthology/N13-1090) 上的例子）。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/linear-relationships.png" alt>
</div>

This explains why these vectors are also useful as features for many canonical
NLP prediction tasks, such as part-of-speech tagging or named entity recognition
(see for example the original work by
[Collobert et al., 2011](https://arxiv.org/abs/1103.0398)
([pdf](https://arxiv.org/pdf/1103.0398.pdf)), or follow-up work by
[Turian et al., 2010](https://www.aclweb.org/anthology/P10-1040)).
这就解释了为什么对于很多典型的 NLP 预测问题来说
这些作为特征的向量都是非常有用的，比如词性标记，指定物体的识别
（可以查看 [Collobert et al., 2011](https://arxiv.org/abs/1103.0398)
([pdf](https://arxiv.org/pdf/1103.0398.pdf)) 的原始工作,
或者 
[Turian et al., 2010](https://www.aclweb.org/anthology/P10-1040) 的追踪任务）。


But for now, let's just use them to draw pretty pictures!
但是现在让我们通过画漂亮的图片来使用它们吧！

## Building the Graph
## 创建 Graph

This is all about embeddings, so let's define our embedding matrix.
This is just a big random matrix to start.  We'll initialize the values to be
uniform in the unit cube.
这就是嵌入的全部，让我们来定义我们的嵌入矩阵吧。
开始这就是一个大型的随机矩阵。然后我们会
给每个单元格初始化一个均匀分布的值。


```python
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
```

The noise-contrastive estimation loss is defined in terms of a logistic regression
model. For this, we need to define the weights and biases for each word in the
vocabulary (also called the `output weights` as opposed to the `input
embeddings`). So let's define that.
对比噪音的估计损失是根据逻辑回归模型来定义的。
这里我们需要为词汇表中的每一个单词定义
权重和 biases（也
叫作 `input embeddings` 的对立 `output weights`）。好，让我们来定义它吧。

```python
nce_weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```

Now that we have the parameters in place, we can define our skip-gram model
graph. For simplicity, let's suppose we've already integerized our text corpus
with a vocabulary so that each word is represented as an integer (see
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
for the details). The skip-gram model takes two inputs. One is a batch full of
integers representing the source context words, the other is for the target
words. Let's create placeholder nodes for these inputs, so that we can feed in
data later.
现在我们有了参数，然后我们就可以定义我们的 skip-gram 模型的 graph 了。
简便起见，假设我们已经用一个词汇表对我们的文本库整数化了，
这样的话每一个单词都是用一个整数来代替的
（更多详细信息请查看 [tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)）。
skip-gram 需要两个输入。一个是代表原上下文单词的全部整数，
另一个是目标单词。让我们为这些输入创建一些占位符节点，这样我们
就可以在之后供给数据了。

```python
# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
```

Now what we need to do is look up the vector for each of the source words in
the batch.  TensorFlow has handy helpers that make this easy.
现在我们需要做的就是分批查询每一个源单词的向量。
使用 TensorFlow 的辅助函数非常简单。

```python
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```

Ok, now that we have the embeddings for each word, we'd like to try to predict
the target word using the noise-contrastive training objective.
好的，现在我们已经拥有了每一个的单词的词向量，让我们
使用噪音对比的训练目标来试着预测一下目标单词。

```python
# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size))
```

Now that we have a loss node, we need to add the nodes required to compute
gradients and update the parameters, etc. For this we will use stochastic
gradient descent, and TensorFlow has handy helpers to make this easy as well.
现在我们得到了一个损失节点，我们需要将必要的节点加入到计算梯度和更新参数中去。
因此我们使用随机提督下降法，同样，使用 TensorFlow 的辅助函数就好了。

```python
# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
```

## Training the Model
## 训练模型

Training the model is then as simple as using a `feed_dict` to push data into
the placeholders and calling
@{tf.Session.run} with this new data
in a loop.
训练模型现在就非常简单了，只需要使用一个 `feed_dict` 将数据推送到 placeholders 中，
并且在一个循环中
使用这个新的数据
来调用 @{tf.Session.run} 就可以了。

```python
for inputs, labels in generate_batch(...):
  feed_dict = {train_inputs: inputs, train_labels: labels}
  _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
```

See the full example code in
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py).
完整实例代码请
查看 [tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)。

## Visualizing the Learned Embeddings
## 可视化训练之后的词向量

After training has finished we can visualize the learned embeddings using
t-SNE.
训练完成之后，我们可以使用 t-SNE 来
可视化训练之后的词向量。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/tsne.png" alt>
</div>

Et voila! As expected, words that are similar end up clustering nearby each
other. For a more heavyweight implementation of word2vec that showcases more of
the advanced features of TensorFlow, see the implementation in
[models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py).
完美！和我们想的一样，同样结尾的单词互相之间离的很近。
还有一个更加重量级的 word2vec 的实现，它会展现更多 TensorFlow 的
高级特性，
请查看 [models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py)。

## Evaluating Embeddings: Analogical Reasoning
## 评估词向量：Analogical Reasoning

Embeddings are useful for a wide variety of prediction tasks in NLP. Short of
training a full-blown part-of-speech model or named-entity model, one simple way
to evaluate embeddings is to directly use them to predict syntactic and semantic
relationships like `king is to queen as father is to ?`. This is called
*analogical reasoning* and the task was introduced by
[Mikolov and colleagues
](https://www.aclweb.org/anthology/N13-1090).
Download the dataset for this task from
[download.tensorflow.org](http://download.tensorflow.org/data/questions-words.txt).
词向量在各种各样的 NLP 预测任务中都非常有用。除了训练一个完善的
词性标记模型或者实体命名模型，一个简单的评估词向量的方法是
直接使用它们来预测语法和语义关系，比如这个句子 `king is to queen as father is to ?`。
这
就是 *analogical reasoning*，这个任务的介绍
在 [Mikolov and colleagues
](https://www.aclweb.org/anthology/N13-1090)。
你可以从 [download.tensorflow.org](http://download.tensorflow.org/data/questions-words.txt) 下载
到这个任务的数据集。

To see how we do this evaluation, have a look at the `build_eval_graph()` and
`eval()` functions in
[models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py).
想看如何实现这个评估过程，
请在 [models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py) 查
看 `build_eval_graph()` 和 `eval()` 这两个函数。

The choice of hyperparameters can strongly influence the accuracy on this task.
To achieve state-of-the-art performance on this task requires training over a
very large dataset, carefully tuning the hyperparameters and making use of
tricks like subsampling the data, which is out of the scope of this tutorial.
超参数的选择会显著的影响到任务的准确性。
为了达到这个任务当前水平的的表现，我们需要使用
一个很大的数据集来训练，并且精细的调优这些超参数以及利用
像二次抽象这样的小技巧，不过二次抽样不在我们的教程范围内。

## Optimizing the Implementation
## 优化实现

Our vanilla implementation showcases the flexibility of TensorFlow. For
example, changing the training objective is as simple as swapping out the call
to `tf.nn.nce_loss()` for an off-the-shelf alternative such as
`tf.nn.sampled_softmax_loss()`. If you have a new idea for a loss function, you
can manually write an expression for the new objective in TensorFlow and let
the optimizer compute its derivatives. This flexibility is invaluable in the
exploratory phase of machine learning model development, where we are trying
out several different ideas and iterating quickly.
我们朴实无华的实现展现了 TensorFlow 的灵活性。
例如，如果我们想改变训练的目标，仅仅通过调用 `tf.nn.nce_loss()` 就可以
使用一个现成的
像 `tf.nn.sampled_softmax_loss()` 这样的函数。如果你对损失函数有一些新的想法，
那么你可以在 TensorFlow 中手动写一个新的目标函数，并且让
优化器计算它的导数。在机器学习模型开发的探索阶段
灵活性是非常重要的，因为这样我们才可以
快速迭代以及实验不同的想法。


Once you have a model structure you're satisfied with, it may be worth
optimizing your implementation to run more efficiently (and cover more data in
less time).  For example, the naive code we used in this tutorial would suffer
compromised speed because we use Python for reading and feeding data items --
each of which require very little work on the TensorFlow back-end.  If you find
your model is seriously bottlenecked on input data, you may want to implement a
custom data reader for your problem, as described in
@{$new_data_formats$New Data Formats}.  For the case of Skip-Gram
modeling, we've actually already done this for you as an example in
[models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py).
一旦你有了一个满意的模型结构，优化你的
实现方法来让让模型更有效率（以及在更少的时间内囊括更多的数据）
就会变得很有必要。举个例子，在这篇教程中使用的本地代码会非常拖后腿，
因为我们使用 Python 来读取和供给数据条目 -- 
在 TensorFlow 这里每一条数据都只需要很少的工作量。如果你发现
你的模型在输入数据方面有很大的性能瓶颈，那么你就需要
为你的问题实现一个自定义的数据读取器，
就像 @{$new_data_formats$New Data Formats} 中说的那样。对于 Skip-Gram 来说，
我们已经为了做了这些，请查看 
[models/tutorials/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec.py)。

If your model is no longer I/O bound but you want still more performance, you
can take things further by writing your own TensorFlow Ops, as described in
@{$adding_an_op$Adding a New Op}.  Again we've provided an
example of this for the Skip-Gram case
[models/tutorials/embedding/word2vec_optimized.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec_optimized.py).
Feel free to benchmark these against each other to measure performance
improvements at each stage.
如果你的模型已经没有 I/O 问题，但是你仍然想得到更好的性能，
那么你可以编写你自己的 TensorFlow 操作，
就像 @{$adding_an_op$Adding a New Op} 中描述的那样。
重申一遍，我们已经提供了 Skip-Gram 模型的完整实例 
[models/tutorials/embedding/word2vec_optimized.py](https://www.tensorflow.org/code/tensorflow_models/tutorials/embedding/word2vec_optimized.py)。
可以随意的和其他模型比较
来测量每一阶段性能的提升。

## Conclusion
## 结论

In this tutorial we covered the word2vec model, a computationally efficient
model for learning word embeddings. We motivated why embeddings are useful,
discussed efficient training techniques and showed how to implement all of this
in TensorFlow. Overall, we hope that this has show-cased how TensorFlow affords
you the flexibility you need for early experimentation, and the control you
later need for bespoke optimized implementation.
本篇教程中，我们讲解了 word2vec 模型，一个学习词向量计算高效的模型。
我们解释了为什么词向量是有用的，讨论了高效训练的技术，同时也展示了
如何在 TensorFlow 中实现这些功能。
总的来说，我们希望已经展示了 TensorFlow 是
如何给你们提供早期的实验阶段你们需要的灵活性，以及
后续你们对定制化优化实现的管理需求。
