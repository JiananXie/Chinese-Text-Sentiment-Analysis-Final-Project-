from Process import get_balance_corpus_2, get_balance_corpus_3, get_balance_corpus_6,feature_extract,parameter_info,get_dict
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding,Masking
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import seaborn as sns   
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.nn.functional import nll_loss
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.metrics import roc_curve, auc, accuracy_score,confusion_matrix
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt

# 1.读取数据
pd_all = pd.read_csv('Sina.csv')
pd_all.dropna(inplace= True)

#是否预处理

# pd_all = feature_extract(pd_all)

#计算LSTM配置参数
vocab_size,max_length = parameter_info(pd_all)

pd_neutral = pd_all[pd_all.label==0]
pd_like = pd_all[pd_all.label==1]
pd_sad = pd_all[pd_all.label==2]
pd_disgust = pd_all[pd_all.label==3]
pd_angry = pd_all[pd_all.label==4]
pd_happy = pd_all[pd_all.label==5]

data = get_balance_corpus_6(36000, pd_neutral, pd_like,pd_sad,pd_disgust,pd_angry,pd_happy)

data_train, data_test = np.split(data.sample(frac=1,random_state=42) ,[int(0.9*len(data))],axis=0)

X_train = data_train['review'].astype(str)
y_train = data_train['label'].values
X_test = data_test['review'].astype(str)
y_test = data_test['label'].values

# 参数设置
embedding_dim = 64  # 嵌入层输出维度
num_classes = 6  # 分类类别数

# 创建Tokenizer对象
tokenizer = Tokenizer(num_words=vocab_size)

# 构建词汇表并将训练数据转换为整数序列
tokenizer.fit_on_texts(X_train)

# 将训练数据和测试数据转换为整数序列
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# 填充序列到相同的长度
X_train_padded = pad_sequences(sequences=X_train_seq, maxlen=max_length,padding='post')
X_train_padded = X_train_padded.reshape(X_train_padded.shape[0],X_train_padded.shape[1],1)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length,padding='post')
X_test_padded = X_test_padded.reshape(X_test_padded.shape[0],X_test_padded.shape[1],1)


# 创建模型
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length,mask_zero=True))
model.add(LSTM(units=128,activation='softmax',dropout=0.2, input_shape=(max_length,1)))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=0.01)
# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train_padded, y_train, epochs=20, batch_size=16, validation_split=0.2,verbose=0)
#保存模型
model.save('LSTM_without_6.h5')
# 评估模型
loss_test, acc_test = model.evaluate(X_test_padded, y_test)
y_prob = model.predict(X_test_padded)
y_pred = np.argmax(y_prob, axis=1)

label_dict = get_dict(num_classes)

nll_test = nll_loss(torch.log_softmax(torch.tensor(y_prob), 1), torch.tensor(y_test))
f1_test = f1_score(y_test, y_pred, average='weighted')
kappa_test = cohen_kappa_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues',xticklabels=label_dict.values(),yticklabels=label_dict.values())
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('cm_LSTM_without_6.png')

num_classes = len(label_dict)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Binarize the labels
if num_classes == 2:
    labels_binary = np.zeros((y_test.shape[0], 2))
    labels_binary[np.arange(y_test.shape[0]), y_test] = 1
else:
    labels_binary = label_binarize(y_test, classes=[i for i in range(num_classes)])

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_binary[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label='{0} (area = {1:0.2f})'.format(label_dict[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_LSTM_without_6.png')

print(
    f'''Test Accuracy: {acc_test: .3f},
    F1-score: {f1_test:.3f},
    NLL: {nll_test:.3f},
    CK: {kappa_test:.3f}''')
