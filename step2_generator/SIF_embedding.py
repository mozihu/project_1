import data_io, params, SIF_core
import os
import numpy as np
from gensim.models import Word2Vec

'''
动态获取词向量
model_100 = Word2Vec.load(os.path.join('/media/brx/TOSHIBA EXT/wiki_zh_word2vec/', 'ngram_100_5_90w.bin'))
words = {}
for index, word in enumerate(model_100.wv.index2entity):
    words[word] = index
We = model_100.wv.vectors
'''

# input
wordfile = '../newsif/datafile/without_stopwords/word2vec_format.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../newsif/datafile/without_stopwords/words_count.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
sentences_test = ['会议充分肯定了2019年金融市场和信贷政策工作取得的成绩，在推动金融市场规范、创新、发展、开放，加大金融支持国家战略和重点领域。', '民营小微企业、精准扶贫力度，稳妥开展互联网金融风险专项整治以及房地产金融宏观审慎管理等方面做了大量卓有成效的工作。', '为实施稳健货币政策、防范化解重大金融风险、推动经济结构调整和转型升级提供了有力支撑。']


# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences
x, m = data_io.sentences2idx(sentences_test, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights

# set parameters
params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_core.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
print(embedding)
