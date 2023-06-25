import sys
sys.path.append('../..')

# 加载数据集
from paddlenlp.datasets import load_dataset
# 飞桨的NLP库
import paddlenlp as ppnlp
# 数据预处理
from utils.utils import prepare_train_features, prepare_validation_featuresf, evaluate
# 偏函数应用
from functools import partial
# 评估SQuAD任务
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
# 集合类
import collections
# 操作时间的函数
import time
# 处理JSON数据
import json

import paddle
from paddlenlp.data import Stack, Dict, Pad

# 加载dureader_robust数据集的训练集和验证集
train_ds, dev_ds = ppnlp.datasets.load_dataset('dureader_robust', splits=('train', 'dev'))

# 设置模型名称
MODEL_NAME = "bert-base-chinese"
# 使用指定模型名称初始化tokenizer
tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(MODEL_NAME)

# 设置最大序列长度
max_seq_length = 512
# 设置文档滑动窗口的步长
doc_stride = 128

# 创建训练集转换函数，使用部分参数和tokenizer
train_trans_func = partial(prepare_train_features,
                           max_seq_length=max_seq_length,
                           doc_stride=doc_stride,
                           tokenizer=tokenizer)

# 对训练集进行转换，并进行批处理
train_ds.map(train_trans_func, batched=True)

# 创建验证集转换函数，使用部分参数和tokenizer
dev_trans_func = partial(prepare_validation_features,
                         max_seq_length=max_seq_length,
                         doc_stride=doc_stride,
                         tokenizer=tokenizer)

# 对验证集进行转换，并进行批处理
dev_ds.map(dev_trans_func, batched=True)

batch_size = 8

# 创建训练集的分布式批量采样器
train_batch_sampler = paddle.io.DistributedBatchSampler(
    train_ds, batch_size=batch_size, shuffle=True)

# 定义训练集的批处理函数
train_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "start_positions": Stack(dtype="int64"),
    "end_positions": Stack(dtype="int64")
}): fn(samples)

# 创建训练集的数据加载器
train_data_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_sampler=train_batch_sampler,
    collate_fn=train_batchify_fn,
    return_list=True)

# 创建验证集的批采样器
dev_batch_sampler = paddle.io.BatchSampler(
    dev_ds, batch_size=batch_size, shuffle=False)

# 定义验证集的批处理函数
dev_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
    "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
}): fn(samples)

# 创建验证集的数据加载器
dev_data_loader = paddle.io.DataLoader(
    dataset=dev_ds,
    batch_sampler=dev_batch_sampler,
    collate_fn=dev_batchify_fn,
    return_list=True)

# 设置想要使用模型的名称
model = ppnlp.transformers.BertForQuestionAnswering.from_pretrained(MODEL_NAME)

# 训练过程中的最大学习率
learning_rate = 3e-5
# 训练轮次
epochs = 1
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

# 计算总的训练步数
num_training_steps = len(train_data_loader) * epochs

# 创建学习率调度器，使用线性衰减和预热
lr_scheduler = ppnlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

# 生成需要进行权重衰减的参数名称列表，排除所有bias和LayerNorm参数
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 创建AdamW优化器，设置学习率、参数和权重衰减
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params)

# 训练过程中的最大学习率
learning_rate = 3e-5
# 训练轮次
epochs = 1
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

# 计算总的训练步数
num_training_steps = len(train_data_loader) * epochs

# 创建学习率调度器，使用线性衰减和预热
lr_scheduler = ppnlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

# 生成需要进行权重衰减的参数名称列表，排除所有bias和LayerNorm参数
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 创建AdamW优化器，设置学习率、参数和权重衰减
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params)

criterion = CrossEntropyLossForSQuAD()
global_step = 0

# 遍历每个训练轮次
for epoch in range(1, epochs + 1):
    # 遍历训练集中的每个批次数据
    for step, batch in enumerate(train_data_loader, start=1):
        global_step += 1
        input_ids, segment_ids, start_positions, end_positions = batch
        logits = model(input_ids=input_ids, token_type_ids=segment_ids)
        loss = criterion(logits, (start_positions, end_positions))

        if global_step % 100 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

    # 在每个轮次结束后进行验证集评估
    evaluate(model=model, data_loader=dev_data_loader)

# 保存模型和tokenizer
model.save_pretrained('./checkpoint')
tokenizer.save_pretrained('./checkpoint')