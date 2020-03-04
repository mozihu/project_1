def knn_Smooth(sentence_list:list,vec_dist:dict)->dict:
    is_begin=True
    dict_smooth = {}
    for i,sentence in enumerate(sentence_list):
        if is_begin==True and i<len(sentence_list)-1:
            is_begin=False
            ###计算平滑后的相似度
            res_value = knn_caculate(main_sentence=sentence,seta1=vec_dist[sentence],after_sentence=sentence_list[i+1],seta3=vec_dist[sentence_list[i+1]])
        elif i== len(sentence_list)-1 and i>0:
            ###计算最后一个的平滑度
            res_value = knn_caculate(main_sentence=sentence,seta1=vec_dist[sentence],pre_sentence=sentence_list[i-1],seta2=vec_dist[sentence_list[i-1]])
        elif 0<i<len(sentence_list)-1:
            #计算中间值的平滑度。
            res_value = knn_caculate(main_sentence=sentence,seta1=vec_dist[sentence],pre_sentence=sentence_list[i-1],seta2=vec_dist[sentence_list[i-1]],after_sentence=sentence_list[i+1],seta3=vec_dist[sentence_list[i+1]])
        dict_smooth[sentence] = res_value
    return dict_smooth

def knn_caculate(main_sentence,seta1,pre_sentence='',seta2,after_sentence='',seta3):
    lens = len(main_sentence)+len(pre_sentence)+len(after_sentence)
    res_value = (1.0*len(main_sentence)*seta1+1.0*len(pre_sentence)*seta2+1.0*len(after_sentence)*seta3)/lens
    return res_value

