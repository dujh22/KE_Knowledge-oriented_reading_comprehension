# 特定目标的容器，以提供Python的内置通用容器dict、list、set、tuple的替代选择
import collections

# 与时间相关的功能
import time

# 使用JSON数据
import json

# PaddlePaddle深度学习平台
import paddle

# 用于评估squad（Question Answering）任务性能的函数
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction

# PaddlePaddle中的装饰器，用于指定该函数下的操作不需要梯度
@paddle.no_grad()
def evaluate(model, data_loader, is_test=False):
    # 将模型设置为评估模式，关闭Dropout和BatchNorm等
    model.eval()

    # 初始化start和end位置的预测结果列表
    all_start_logits = []
    all_end_logits = []
    # 获取当前时间，用于计算每1000个样本的处理时间
    tic_eval = time.time()

    # 遍历数据集中的每个batch
    for batch in data_loader:
        # 从batch中提取input_ids和token_type_ids
        input_ids, token_type_ids = batch
        # 将input_ids和token_type_ids输入模型，获取开始和结束位置的预测结果
        start_logits_tensor, end_logits_tensor = model(input_ids, token_type_ids)

        # 遍历每个样本的预测结果
        for idx in range(start_logits_tensor.shape[0]):
            # 每处理1000个样本，打印处理进度和处理时间
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            # 将每个样本的预测结果添加到预测结果列表中
            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    # 使用预测结果生成最终的答案
    all_predictions, _, _ = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), False, 20, 30)

    # 如果是测试阶段
    if is_test:
        # 将预测结果写入JSON文件中
        with open('prediction.json', "w", encoding='utf-8') as writer:
            writer.write(
                json.dumps(
                    all_predictions, ensure_ascii=False, indent=4) + "\n")
    else:
        # 如果不是测试阶段，则对预测结果进行评估
        squad_evaluate(
            examples=data_loader.dataset.data,
            preds=all_predictions,
            is_whitespace_splited=False)

    # 打印前5个问题和预测的答案
    count = 0
    for example in data_loader.dataset.data:
        count += 1
        print()
        print('问题：', example['question'])
        print('原文：', ''.join(example['context']))
        print('答案：', all_predictions[example['id']])
        if count >= 5:
            break

    # 将模型设置回训练模式
    model.train()

def prepare_train_features(examples, tokenizer, doc_stride, max_seq_length):
    # 通过提取上下文和问题创建问题和上下文的列表
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    # 使用tokenizer对问题和上下文进行编码，同时使用stride处理长文本的溢出
    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length)

    return_examples = []
    temp_example = {}
    # 为每个编码后的样本添加标签
    for i in range(len(tokenized_examples["input_ids"])):
        # 对于不可能的答案，我们将其标签设为CLS token的索引
        input_ids = tokenized_examples["input_ids"][i]
        temp_example["input_ids"] = input_ids
        cls_index = input_ids.index(tokenizer.cls_token_id)
    
        # 偏移映射可以将token映射到原始上下文中的字符位置，这对于计算开始和结束位置非常有用
        offsets = tokenized_examples['offset_mapping'][i]
        temp_example["offset_mapping"] = offsets
    
        # 获取与该样本对应的序列（以了解哪部分是上下文，哪部分是问题）
        sequence_ids = tokenized_examples['token_type_ids'][i]
        temp_example["token_type_ids"] = sequence_ids
    
        # 一个样本可能给出多个span，此处是包含该文本span的样本的索引
        sample_index = tokenized_examples['overflow_to_sample'][i]
        temp_example["overflow_to_sample"] = sample_index
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']
    
        # 答案在文本中的开始和结束字符索引
        start_char = answer_starts[0]
        end_char = start_char + len(answers[0])
    
        # 当前span在文本中的开始token索引
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
    
        # 当前span在文本中的结束token索引
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
        # 再减一以达到实际文本
        token_end_index -= 1
    
        # 检查答案是否在span外（如果在span外，这个特性就用CLS索引标记）
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            temp_example["start_positions"] = cls_index
            temp_example["end_positions"] = cls_index
        else:
            # 否则，将token_start_index和token_end_index移动到答案的两端
            # 注意：如果答案是最后一个词，我们可能会超过最后的偏移（边缘情况）
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            temp_example["start_positions"] = token_start_index - 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            temp_example["end_positions"] = token_end_index + 1

        return_examples.append(temp_example)

    return return_examples

def prepare_validation_features(examples, tokenizer, doc_stride, max_seq_length):
    # 通过提取上下文和问题创建问题和上下文的列表
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    # 使用tokenizer对问题和上下文进行编码，同时使用stride处理长文本的溢出
    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length)

    return_examples = []
    temp_example = {}
    # 对于验证集，不需要计算开始和结束位置
    for i in range(len(tokenized_examples["token_type_ids"])):
        temp_example["input_ids"] = tokenized_examples["input_ids"][i]

        # 获取与该样本对应的序列（以了解哪部分是上下文，哪部分是问题）
        temp_example["token_type_ids"] = tokenized_examples['token_type_ids'][i]
        sequence_ids = temp_example["token_type_ids"]

        # 一个样本可能给出多个span，此处是包含该文本span的样本的索引
        temp_example["overflow_to_sample"] = tokenized_examples['overflow_to_sample'][i]
        sample_index = temp_example["overflow_to_sample"]

        # 保存该样本的id
        temp_example["example_id"] = examples[sample_index]['id']

        # 将不属于上下文的offset_mapping设为None，方便确定token位置是否为上下文的一部分
        temp2_example = tokenized_examples['offset_mapping'][i]
        temp_example["offset_mapping"] = [(o if sequence_ids[k] == 1 else None) for k, o in enumerate(temp2_example)]

        return_examples.append(temp_example)

    return return_examples

