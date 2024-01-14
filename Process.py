import pandas as pd
import jieba
import jieba.posseg as seg

def get_dict(num_labels):
        if(num_labels == 2):
            return {0: "Negative", 1: "Positive"}
        elif(num_labels == 3):
            return {0: "Negative", 1: "Neutral", 2: "Positive"}
        else:
            return {0: "Neutral", 1: "Like", 2: "Sad", 3: "Disgust", 4: "Angry", 5: "Happy"}



def get_balance_corpus_2(corpus_size, corpus_pos, corpus_neg):
    sample_size = corpus_size // 2
    pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0]<sample_size),
                                   corpus_neg.sample(sample_size, replace=corpus_neg.shape[0]<sample_size)])
       
    print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
    print('评论数目（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label==1].shape[0])
    print('评论数目（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label==0].shape[0])    
    
    return pd_corpus_balance

def get_balance_corpus_6(corpus_size, corpus1, corpus2,corpus3,corpus4,corpus5,corpus6):
    sample_size = corpus_size // 6
    pd_corpus_balance = pd.concat([corpus1.sample(sample_size, replace=corpus1.shape[0] < sample_size), 
                                   corpus2.sample(sample_size, replace=corpus2.shape[0] < sample_size),
                                   corpus3.sample(sample_size, replace=corpus3.shape[0] < sample_size),
                                   corpus4.sample(sample_size, replace=corpus4.shape[0] < sample_size),
                                   corpus5.sample(sample_size, replace=corpus5.shape[0] < sample_size),
                                   corpus6.sample(sample_size, replace=corpus6.shape[0] < sample_size)])
    print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
    print('评论数目（高兴）：%d' % pd_corpus_balance[pd_corpus_balance.label==5].shape[0])
    print('评论数目（愤怒）：%d' % pd_corpus_balance[pd_corpus_balance.label==4].shape[0])
    print('评论数目（厌恶）：%d' % pd_corpus_balance[pd_corpus_balance.label==3].shape[0])
    print('评论数目（悲伤）：%d' % pd_corpus_balance[pd_corpus_balance.label==2].shape[0])
    print('评论数目（喜好）：%d' % pd_corpus_balance[pd_corpus_balance.label==1].shape[0])
    print('评论数目（中性）：%d' % pd_corpus_balance[pd_corpus_balance.label==0].shape[0])
    return pd_corpus_balance

def get_balance_corpus_3(corpus_size, corpus_pos, corpus_neg, corpus_neu):
    sample_size = corpus_size // 3
    pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0]<sample_size),
                                   corpus_neg.sample(sample_size, replace=corpus_neg.shape[0]<sample_size),
                                   corpus_neu.sample(sample_size, replace=corpus_neu.shape[0]<sample_size)])     
       
    print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
    print('评论数目（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label==2].shape[0])
    print('评论数目（中性）：%d' % pd_corpus_balance[pd_corpus_balance.label==1].shape[0])
    print('评论数目（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label==0].shape[0])    
    
    return pd_corpus_balance

def feature_extract(data):
    feature_type = ['a','an','av','v','z','d','c']
    data['review'] = data['review'].astype(str).apply(lambda x: ' '.join([w.word for w in seg.lcut(x,use_paddle=True) if w.flag in feature_type]))
    return data

def parameter_info(data):
    word_set = set()
    for review in data['review'].astype(str):
        words = review.split()
        word_set.update(words)
    data['len'] = data['review'].astype(str).apply(lambda x: len(x))
    word_set_size = len(word_set)
    print('单词表大小：%d' % word_set_size)
    print('字符串0.9分位：%d' % int(data['len'].quantile(0.90)))

    return word_set_size, int(data['len'].quantile(0.90))

