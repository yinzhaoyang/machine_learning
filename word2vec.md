就是将词(word)转化成N维的数字(通常为float型)向量。

词向量有两种表示：<br/>
**one-hot representation**<br/>
把所有的词按许排列，N个词就用N维向量表示，某个词对应位置1，其他位置为0的向量即为这个词的向量表示,如[00001000]。
但这种词表示有两个缺点：（1）容易受维数灾难的困扰，尤其是将其用于 Deep Learning 的一些算法时；（2）不能很好地刻画词与词之间的相似性（术语好像叫做“词汇鸿沟”）：任意两个词之间都是孤立的。光从这两个向量中看不出两个词是否有关系，哪怕是话筒和麦克这样的同义词也不能幸免于难。

**distributed representation**<br/>
用一个固定维度N(50,100,200等)的向量表示一个词集中的各个词。相当于将词集中映射到一个N维向量空间，空间中的每个点表示一个词。量点(词向量)之间的距离就是两个词之间的语法语义的相似性。
把超过N个词的词集映射到N为向量空间，需要对词集进行处理，word2vec是其中的一种方式。

**语言模型**<br/>
计算词序列s是一个合规的句子的概率：

N-gram模型  ----

N-pos模型  ---- 先将V个词映射到K个类别，考虑w在类别上下文中的概率。


skip-gram: 根据current-word 预测neighbors(context)
CBOW:根据context预测current word

例如，对于10000个词集，每个词的输入one-hot representation的1*10000向量(词对应位置为1，其他为0)。隐藏层为10000*300的矩阵，每个词的distributed representation为1*300的向量。输出层的转化算法将词X的distributed representation向量计算得出词集中每个词在X上下文中的概率。

The hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron)
![](https://github.com/yinzhaoyang/machine_learning/blob/master/word2vec_nn.png)
![](https://github.com/yinzhaoyang/machine_learning/blob/master/word2vec_hidden.png)
![](https://github.com/yinzhaoyang/machine_learning/blob/master/word2vec_output.png)

**参考：**
word2vec-resources<br/>
http://mccormickml.com/2016/04/27/word2vec-resources/
https://www.tensorflow.org/tutorials/word2vec

word2vec tutorial<br/>
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

word2vec model  paper<br/>
http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

word2vec implementation with TF<br/>
https://github.com/mchablani/deep-learning/blob/master/embeddings/Skip-Gram_word2vec.ipynb
