import paddle
import json
import re
from paddle.io import Dataset, DataLoader
from paddlenlp.datasets import MapDataset

# 加载原始数据集
way = '/home/bignet/code/knowledgeEngineer/train.json'
temp_name1 = "KoRC_H"
temp_name2 = "KoRC-H"

dataset = []
with open(way, 'r', encoding='utf-8') as f:
    d_list = json.load(f)
    for item in d_list:
        id = item['id']
        title = item['title']
        context = item['passage']
        question = item['question'][temp_name2]
        answers = []
        answers.append(item['answers'][0])
        answer_starts = []
        try:
            question_s = (re.findall(r'\[.*?\]', context)[0]).strip("[]")
            answer_starts_temp = context.find(question_s)
        except:
            answer_starts_temp = -1
        answer_starts.append(answer_starts_temp)
        dataset.append({'id': id, 'title': title, 'context': context, 'question': question, 'answers': answers, 'answer_starts': answer_starts})

# 构建 MapDataset
train_ds = MapDataset(dataset)

# 循环打印前两个训练样本的相关信息
for idx in range(2):
    # 打印问题
    print(train_ds[idx]['question'])
    # 打印问题的上下文
    print(train_ds[idx]['context'])
    # 打印问题的答案
    print(train_ds[idx]['answers'])
    # 打印答案在上下文中的开始位置
    print(train_ds[idx]['answer_starts'])
    # 打印一个空行，为了在控制台输出中区分各个样本
    print()

# 加载原始数据集
way2 = '/home/bignet/code/knowledgeEngineer/valid.json'

dataset2 = []
with open(way2, 'r', encoding='utf-8') as f:
    d_list = json.load(f)
    for item in d_list:
        id = item['id']
        title = item['title']
        context = item['passage']
        question = item['question'][temp_name2]
        answers = []
        answers.append(item['answers'][0])
        answer_starts = []
        try:
            question_s = (re.findall(r'\[.*?\]', context)[0]).strip("[]")
            answer_starts_temp = context.find(question_s)
        except:
            answer_starts_temp = -1
        answer_starts.append(answer_starts_temp)
        dataset2.append({'id': id, 'title': title, 'context': context, 'question': question, 'answers': answers, 'answer_starts': answer_starts})

# 构建 MapDataset
dev_ds = MapDataset(dataset2)

# 循环打印前两个验证样本的相关信息
for idx in range(2):
    # 打印问题
    print(dev_ds[idx]['question'])
    # 打印问题的上下文
    print(dev_ds[idx]['context'])
    # 打印问题的答案
    print(dev_ds[idx]['answers'])
    # 打印答案在上下文中的开始位置
    print(dev_ds[idx]['answer_starts'])
    # 打印一个空行，为了在控制台输出中区分各个样本
    print()