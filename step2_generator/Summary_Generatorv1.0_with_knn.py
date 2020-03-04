'''
算法变动：在v1.0的基础上对生成后的句向量进行KNN平滑计算
'''


import re
import SIF_core, data_io, params
import numpy as np
import os
from gensim.models import Word2Vec




fulltext = []
while True:
    try:
        fulltext.append(input('请输入目标文本：'))
    except:
        break
#fulltext = input('请输入目标文章全文：')
fulltext = ''.join(fulltext)
print('fulltext:',fulltext)
print('type of fulltext:', type(fulltext))
#fulltext = fulltext.split()
# 输入文章及标题
title = input('请输入目标文章的标题：')
title = title.strip()
title = ''.join(title.split())
# 将文章按照汉语结束标点切分成句子（生成器）
def cuto_sentences(article):
    if not isinstance(article, str):
        article = str(article)
    puns = frozenset(u'。！？；，')# 在第一版本的基础上增加逗号分隔，因为后续的KNN平滑需要考虑短句子。
    tmp = []
    for ch in article:
        tmp.append(ch)
        if puns.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)
# 将生成的句子放入列表待用
def article_sents(article):
    article = article.strip()
    sentences = []
    for i in cuto_sentences(article):
        if i:
            sentences.append(i.strip())
    return sentences

# 词向量文件，词频文件，超参数设置
wordfile = '../newsif/datafile/without_stopwords/word2vec_format.txt'
weightfile = '../newsif/datafile/without_stopwords/words_count.txt'
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme

# 生成句向量的函数
def get_sent_vec(sentences):
    import params
    # 详见data_io.py
    (words, We) = data_io.getWordmap(wordfile)
    # 详见data_io.py
    word2weight = data_io.getWordWeight(weightfile, weightpara)
    weight4ind = data_io.getWeight(words, word2weight)
    # 详见data_io.py
    x, m = data_io.sentences2idx(sentences, words)
    w = data_io.seq2weight(x, m, weight4ind)

    # 参数设置
    params = params.params()
    params.rmpc = rmpc
    # 调用SIF核心算法计算句向量，详见SIF_core
    embedding = SIF_core.SIF_embedding(We, x, w, params)

    get_sent_vec = {}
    for i in range(len(embedding)):
        get_sent_vec[sentences[i]] = embedding[i]

    return get_sent_vec

# 处理文章，分别计算全文向量，句向量，标题向量
articleTosents = article_sents(fulltext)
print('articleTosents:',articleTosents)# 调试用
Vsj = get_sent_vec(articleTosents)
#print('Vsj[articleTosents]:',Vsj)

#全文向量
wholearticle = ''.join(fulltext.split())
#print('wholearticle:',wholearticle)
#print('type of wholearticle:',type(wholearticle))
Vc = get_sent_vec(wholearticle.split())
#print('Vc[wholearticle]:',Vc)
dVc = Vc[wholearticle].tolist()
#print('dVc:',dVc)

#标题向量
#print('title:',title)
Vt = get_sent_vec(title.split())
#print('Vt[title]:',Vt)
dVt = Vt[title].tolist()
#print('dVt:',dVt)

#计算句向量余弦距离的函数
def get_dist(v1, v2):
    get_dist = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return get_dist

#分别计算句向量中每一句与全文向量和标题向量的余弦距离，存为字典
vec_dist1 = {}
vec_dist2 = {}

#计算过程（有bug！！！！）
for key in Vsj:
    dVsj = Vsj[key].tolist()
    dist1 = get_dist(dVsj,dVc)
    vec_dist1[key] = dist1
    dist2 = get_dist(dVsj, dVt)
    vec_dist2[key] = dist2

#生成摘要的函数用到的超参数
a = 0.8
t = 0.2
#计算句向量与全文向量和标题向量的加权值，用来判断句向量与全文和标题的近似成都
vec_dist = {}
for key in Vsj:
    dist = vec_dist1[key] * a + vec_dist2[key] * t
    vec_dist[key] = dist

#调用KNN计算方法对已经得到的句向量进行前后平滑，返回平滑后句向量的字典（句子：句向量）
def knn_Smooth(sentence_list:list,vec_dis:dict)->dict:
    is_begin=True
    dict_smooth = {}
    for i,sentence in enumerate(sentence_list):
        if is_begin==True and i<len(sentence_list)-1:
            is_begin=False
            ###计算平滑后的相似度
            res_value = knn_caculate(main_sentence=sentence,seta1=vec_dis[sentence],after_sentence=sentence_list[i+1],seta3=vec_dis[sentence_list[i+1]])
        elif i== len(sentence_list)-1 and i>0:
            ###计算最后一个的平滑度
            res_value = knn_caculate(main_sentence=sentence,seta1=vec_dis[sentence],pre_sentence=sentence_list[i-1],seta2=vec_dis[sentence_list[i-1]])
        elif 0<i<len(sentence_list)-1:
            #计算中间值的平滑度。
            res_value = knn_caculate(main_sentence=sentence,seta1=vec_dis[sentence],pre_sentence=sentence_list[i-1],seta2=vec_dis[sentence_list[i-1]],after_sentence=sentence_list[i+1],seta3=vec_dis[sentence_list[i+1]])
        dict_smooth[sentence] = res_value
    return dict_smooth

#定义KNN计算方法
def knn_caculate (main_sentence,seta1,pre_sentence='',seta2=0,after_sentence='',seta3=0):
    lens = len(main_sentence)+len(pre_sentence)+len(after_sentence)
    res_value = (1.0*len(main_sentence)*seta1+1.0*len(pre_sentence)*seta2+1.0*len(after_sentence)*seta3)/lens
    return res_value

#对目标文章的句向量vec_dist进行KNN平滑
smooth_dist = {}
for i in articleTosents:
    smooth_dist[i] = knn_Smooth(articleTosents, vec_dist)[i]
print(smooth_dist)

#排序并取出近似度最近的5句话
res = sorted(smooth_dist.items(), key=lambda d: d[1], reverse=True)
#print(res)
#print(type(res[1][0]))
result = ''
#考虑生成摘要句子的句子顺序是否会影响，标记在原文的顺序
for i in range(8):
    print(res[i][0])
    result += res[i][0]

#输出摘要文章
print('参考摘要为：',result)


    


