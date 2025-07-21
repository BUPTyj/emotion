import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import evaluate

def count_labels(ds):
    label_counts = Counter(ds['label'])
    return label_counts.get(0, 0), label_counts.get(1, 0)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)


acc_metric = evaluate.load("accuracy")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    rec = recall_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(rec)
    acc.update(f1)
    return acc

def extract(metric):
    xs, ys = [], []
    for entry in history:
        if metric in entry and "epoch" in entry:
            xs.append(entry["epoch"])
            ys.append(entry[metric])
    return xs, ys

if __name__ == '__main__':
    dataset = load_dataset("lansinuote/ChnSentiCorp", cache_dir="data")
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    test_dataset = dataset['test']

    # 统计各部分正负样本的数量
    train_negative, train_positive = count_labels(train_dataset)
    valid_negative, valid_positive = count_labels(valid_dataset)
    test_negative, test_positive = count_labels(test_dataset)

    categories = ['Train', 'Validation', 'Test']
    negative_counts = [train_negative, valid_negative, test_negative]
    positive_counts = [train_positive, valid_positive, test_positive]
    bar_width = 0.35
    index = range(len(categories))

    fig, ax = plt.subplots()
    bar1 = ax.bar(index, negative_counts, bar_width, label='Negative', color='red')
    bar2 = ax.bar([i + bar_width for i in index], positive_counts, bar_width, label='Positive', color='blue')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Positive and Negative Samples')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(categories)
    ax.legend()

    plt.show()

    # 加载 BERT tokenizer 和模型
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print("模型加载完成")

    encoded_datasets = dataset.map(preprocess_function, batched=True)
    print("数据预处理完成")

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=3e-5,
        metric_for_best_model="f1",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        load_best_model_at_end=True,
    )

    print("开始训练")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets['train'],
        eval_dataset=encoded_datasets['validation'],
        compute_metrics=eval_metric,
    )

    hyper_params = {
        "模型": model.__class__.__name__,
        "隐藏层大小": getattr(model.config, "hidden_size", "N/A"),
        "训练 epoch": training_args.num_train_epochs,
        "训练 batch_size": training_args.per_device_train_batch_size,
        "验证 batch_size": training_args.per_device_eval_batch_size,
        "学习率": training_args.learning_rate,
        "学习率 warm‑up 步数": training_args.warmup_steps,
        "权重衰减": training_args.weight_decay,
        "优化器": training_args.optim,
        "设备": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    print("\n======== 超参数一览 ========")
    for k, v in hyper_params.items():
        print(f"{k:10}: {v}")
    print("================\n")

    trainer.train()

    history = trainer.state.log_history

    epochs_acc, vals_acc = extract("eval_accuracy")
    epochs_rec, vals_rec = extract("eval_recall")
    epochs_f1 , vals_f1  = extract("eval_f1")
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_acc, vals_acc, label="Accuracy")
    plt.plot(epochs_rec, vals_rec, label="Recall")
    plt.plot(epochs_f1 , vals_f1 , label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Validation metrics during training")
    plt.legend()
    plt.tight_layout()
    plt.show()

    train_metrics = trainer.evaluate(encoded_datasets["train"])
    print(f"训练集 ➜  accuracy: {train_metrics['eval_accuracy']:.4f} | "
          f"recall: {train_metrics['eval_recall']:.4f} | "
          f"f1: {train_metrics['eval_f1']:.4f}")

    test_metrics = trainer.evaluate(encoded_datasets["test"])
    print(f"测试集 ➜  accuracy: {test_metrics['eval_accuracy']:.4f} | "
          f"recall: {test_metrics['eval_recall']:.4f} | "
          f"f1: {test_metrics['eval_f1']:.4f}")

    # 保存模型参数
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
