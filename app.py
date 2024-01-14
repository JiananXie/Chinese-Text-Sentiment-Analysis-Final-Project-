from flask import Flask, request, jsonify
from flask_cors import CORS
from Bert_Sentiment import predict, BertClassifier
import torch


bert_model2 = BertClassifier(num_labels=2)
bert_model2.load_state_dict(torch.load('model_without_2.pth'))
print('二分类BERT加载完成')
bert_model3 = BertClassifier(num_labels=3)
bert_model3.load_state_dict(torch.load('model_without_3.pth'))
print('三分类BERT加载完成')
bert_model6 = BertClassifier(num_labels=6)
bert_model6.load_state_dict(torch.load('model6.pth'))
print('六分类BERT加载完成')
bert = {}
bert[2] = bert_model2
bert[3] = bert_model3
bert[6] = bert_model6


app = Flask(__name__)
CORS(app)  # 这会为所有路由启用CORS

@app.route('/sentiment', methods=['POST'])
def sentiment():
    try:
        data = request.get_json()
        text = data['text']
        num_labels = {'二分类': 2, '三分类': 3, '六分类':6}[data['classificationType']]
        str1=''
        str1+=f'\nBERT的预测结果为：{predict(text, bert[num_labels])}'
        return jsonify({'result': str1})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
@app.route('/test', methods=['GET'])
def test():
    return 'Server is working'

if __name__ == '__main__':
    # init()
    app.run(host='0.0.0.0',port=5000)
