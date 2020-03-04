'''
调整代码课接收文章分段输入；
对每段进行处理，选择每段的概括语句作为全文概括；
数学模型仍使用余弦相似度计算；
增加词云图；
——实现于2020/3/3
实验结果：相关功能可实现，摘要不理想，以百度生成做对比，句子连贯性差，概括性差
下一步计划： 相似度算法，及重点句子选择规则，重新定制

待验证数学模型：句向量聚类模型；textrank算法实现；
'''


import re
import SIF_core, data_io, params
import numpy as np
import os
from gensim.models import Word2Vec
import wordcloud
import matplotlib.pyplot as plt
import jieba
import re
import PIL

# 输入文章及标题
title = input('请输入目标文章的标题：')
title = title.strip()
title = ''.join(title.split())

#输入全文
fulltext_list = []
print('请输入全文内容，按回车键结束：')
while True:
    temp = input()
    if temp == '':
        break
    fulltext_list.append(temp)
fulltext = ''.join(fulltext_list)

#生成词云图
text1 = fulltext
#导入图片
image1 = PIL.Image.open(r'./without_stopwords/blackboard_word_cloud.jpg')
MASK = np.array(image1)
WC = wordcloud.WordCloud(font_path = 'C:\\Windows\\Fonts\\STFANGSO.TTF',max_words=100,mask = MASK,height= 400,width=400,background_color='white',repeat=False,mode='RGBA') #设置词云图对象属性
st1 = re.sub('[，。、“”‘ ’]','',str(text1)) #使用正则表达式将符号替换掉。
conten = ' '.join(jieba.lcut(st1)) #此处分词之间要有空格隔开，联想到英文书写方式，每个单词之间都有一个空格。
con = WC.generate(conten)
plt.imshow(con)
plt.axis('off')
plt.show()
con.to_file('test_word_cloud.png')

# 将文章按照汉语结束标点切分成句子（生成器）
def cuto_sentences(article):
    if not isinstance(article, str):
        article = str(article)
    puns = frozenset(u'。！？；')
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
wordfile = './without_stopwords/word2vec_format.txt'
weightfile = './without_stopwords/words_count.txt'
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


#标题向量
Vt = get_sent_vec(title.split())
dVt = Vt[title].tolist()

#全文向量
wholearticle = ''.join(fulltext.split())
Vc = get_sent_vec(wholearticle.split())
dVc = Vc[wholearticle].tolist()

#计算段向量，并存储到dict
paragraph_vecs = {}
for a in range(len(fulltext_list)):
    paragraph_sent = ''.join(fulltext_list[a].split())
    paragraph_vecs[a] = get_sent_vec(paragraph_sent)[paragraph_sent[0]].tolist()

# 处理文章，分别计算全文向量，句向量，标题向量
articleTosents = article_sents(fulltext)
Vsj = get_sent_vec(articleTosents)

#计算句向量余弦距离的函数
def get_dist(v1, v2):
    get_dist = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return get_dist

#分别计算句向量中每一句与全文向量和标题向量,段落向量的余弦距离，存为字典
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

#排序并取出近似度最近的1句话作为对全文的概括
res_all = sorted(vec_dist.items(), key=lambda d: d[1], reverse=True)

#计算每个句向量到每个段向量的余弦相似度，保存为字典（字典的key值为段落，value值为字典（key值为句子,value值为本句子向量到本段向量的距离))
paragraph_dist = {}
for a in range(len(fulltext_list)):
    vec_dist3 = {}
    for key in Vsj:
        dVsj = Vsj[key].tolist()
        a_vec = paragraph_vecs[a]
        dist_3 = get_dist(dVsj, a_vec)
        vec_dist3[key] = dist_3
    paragraph_dist[a] = vec_dist3
    
#排序并取出近似度最近的1句话作为每段的概括,存为list
res_part = []
for a in range(len(fulltext_list)):
    for key in Vsj:
        sorted_paragraph = sorted(paragraph_dist[a].items(), key=lambda d: d[1], reverse=True)
    res_part.append(sorted_paragraph[0])
print(res_part)
#组合生成最终摘要
result = res_all[0][0]
for a in range(len(res_part)):
    result += res_part[a][0]
print('参考摘要为：', result)

