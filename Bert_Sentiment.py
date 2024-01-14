import pandas as pd
import numpy as np
import torch
import numpy as np
from torch import nn
from transformers import BertTokenizer, BertModel,get_linear_schedule_with_warmup
from torch.optim import Adam
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, cohen_kappa_score
from torch.nn.functional import nll_loss,softmax
from Process import get_balance_corpus_2, get_balance_corpus_3, get_balance_corpus_6,get_dict,feature_extract

tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')

def predict(text,model):
    input = tokenizer(text, 
                      padding='max_length', 
                      max_length = 512, 
                      truncation=True,
                      return_tensors="pt")

    mask = input['attention_mask']
    input_id = input['input_ids'].squeeze(1)

    output = model(input_id, mask)
    prediction = output.argmax(dim=1).item()
    if model.num_labels == 2:
        if prediction == 1:
            prediction = 'Positive'
        else:  
            prediction = 'Negative' 
        return prediction
    elif model.num_labels == 3:
        if prediction == 0:
            prediction = 'Negative'
        elif prediction == 1:
            prediction = 'Neutral'
        else:
            prediction = 'Positive'
    else:
        if prediction == 0:
            prediction = 'Neutral'
        elif prediction == 1:
            prediction = 'Like'
        elif prediction == 2:
            prediction = 'Sad'
        elif prediction == 3:
            prediction = 'Disgust'
        elif prediction == 4:
            prediction = 'Angry'
        else:
            prediction = 'Happy' 
    return prediction

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = df['label'].values
        self.reviews = [tokenizer(review, 
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") 
                      for review in df['review']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_reviews(self, idx):
        # Fetch a batch of inputs
        return self.reviews[idx]

    def __getitem__(self, idx):
        batch_reviews = self.get_batch_reviews(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_reviews, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.4,num_labels=2):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('chinese-bert-wwm-ext')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
  # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
  # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 开始进入训练循环

    min_valid_loss = float('inf')
    patience_counter = 0
    patience_limit = 2
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
            total_acc_train = 0
            total_loss_train = 0
            model.train()

      # 训练集循环 
            for train_input, train_label in train_dataloader:
                train_label = train_label.type(torch.LongTensor).to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通过模型得到输出
                optimizer.zero_grad()
                output = model(input_id, mask)
                # 计算损失
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # 计算精度
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
        # 模型更新
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
            # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_val = 0
            total_loss_val = 0
            model.eval()

      # 不需要计算梯度
            with torch.no_grad():
                # 循环获取数据集，并用训练好的模型进行验证
                for val_input, val_label in val_dataloader:
          # 如果有GPU，则使用GPU，接下来的操作同训练
                    val_label = val_label.type(torch.LongTensor).to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            train_losses.append(total_loss_train / len(train_data))
            val_losses.append(total_loss_val / len(val_data))
            train_accs.append(total_acc_train / len(train_data))
            val_accs.append(total_acc_val / len(val_data))            

            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')   
            if total_loss_val < min_valid_loss:
                min_valid_loss = total_loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience_limit:
                print('Early Stopping')
                break 
    fig, ax1 = plt.subplots()   
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, color=color, label='Train Loss')
    ax1.plot(val_losses, color=color, linestyle='dashed', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim([0, len(train_losses)])
    ax1.set_ylim([0, 1.5])
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(train_accs, color=color, label='Train Acc')
    ax2.plot(val_accs, color=color, linestyle='dashed', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0,1])
    ax2.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('train_val_loss_acc_without_6.png')

def evaluate(model, test_data):
    model.eval()
    test = Dataset(test_data)
    total_acc_test = 0

    #判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if use_cuda:
        model = model.cuda()

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)

    preds = []
    labels = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.type(torch.LongTensor).to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)


            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            preds.extend(softmax(output,dim=1).cpu().numpy())
            labels.extend(test_label.cpu().numpy())
    # 计算F1-score、NLL和Cohen's Kappa系数
    nll = nll_loss(torch.log(torch.tensor(preds)), torch.tensor(labels))

    num_classes = model.num_labels
    label_dict = get_dict(num_classes)


    # Binarize the labels
    if num_classes == 2:
        labels_binary = np.zeros((len(labels), 2))
        labels_binary[np.arange(len(labels)), labels] = 1
    else:
        labels_binary = label_binarize(labels, classes=[i for i in range(num_classes)])
    
    preds = np.array(preds)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_binary[:, i], preds[:, i])
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
    plt.savefig('roc_without_6.png')

    preds = np.argmax(preds,axis=1)
    f1 = f1_score(labels, preds, average='weighted')
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d',cmap='Blues',xticklabels=label_dict.values(),yticklabels=label_dict.values())
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('cm_without_6.png')

    print(
        f'''Test Accuracy: {total_acc_test / len(test_data): .3f},
        F1-score: {f1:.3f},
        NLL: {nll:.3f},
        CK: {kappa:.3f}''')

if __name__ == '__main__':
    pd_all = pd.read_csv('Sina.csv')
    pd_all.dropna(inplace= True)

    pd_neutral = pd_all[pd_all.label==0]
    pd_like = pd_all[pd_all.label==1]
    pd_sad = pd_all[pd_all.label==2]
    pd_disgust = pd_all[pd_all.label==3]
    pd_angry = pd_all[pd_all.label==4]
    pd_happy = pd_all[pd_all.label==5]

    data = get_balance_corpus_6(36000, pd_neutral, pd_like,pd_sad,pd_disgust,pd_angry,pd_happy)

    data = feature_extract(data)
    data_train, data_valid= np.split(data.sample(frac=1,random_state=42) ,[int(0.8*len(data))],axis=0)
    data_valid, data_test = np.split(data_valid.sample(frac=1,random_state=42) ,[int(0.5*len(data_valid))],axis=0)

    EPOCHS = 10
    model = BertClassifier(num_labels=6)
    LR = 1e-6
    train(model, data_train, data_valid, LR, EPOCHS)
    evaluate(model, data_test)
    torch.save(model.state_dict(), 'model_without_6.pth')