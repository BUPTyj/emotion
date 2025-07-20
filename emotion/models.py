from django.db import models
from django.db.models import JSONField


# Create your models here.
class QA(models.Model):
    question = models.TextField(verbose_name="需要分析的字段")
    answer = models.TextField(verbose_name="分析结果")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    is_delete = models.BooleanField(default=False, verbose_name="是否删除")


class TrainingStatus(models.Model):
    # 基本状态字段
    is_training = models.BooleanField(default=False, verbose_name="是否正在训练")
    current_epoch = models.IntegerField(default=0, verbose_name="当前轮数")
    total_epochs = models.IntegerField(default=0, verbose_name="总轮数")
    current_step = models.IntegerField(default=0, verbose_name="当前步数")
    total_steps = models.IntegerField(default=0, verbose_name="总步数")

    # 损失和评估指标
    train_loss = models.FloatField(default=0.0, verbose_name="训练损失")
    eval_loss = models.FloatField(default=0.0, verbose_name="评估损失")
    eval_f1 = models.FloatField(default=0.0, verbose_name="F1分数")
    eval_accuracy = models.FloatField(default=0.0, verbose_name="准确率")
    eval_recall = models.FloatField(default=0.0, verbose_name="召回率")
    progress_percent = models.FloatField(default=0.0, verbose_name="进度百分比")

    # JSON字段存储复杂数据
    logs = JSONField(default=list, blank=True, verbose_name="训练日志")
    final_metrics = JSONField(default=dict, blank=True, verbose_name="最终指标")
    hyperparameters = JSONField(default=dict, blank=True, verbose_name="超参数")

    # 其他字段
    error_message = models.TextField(blank=True, default='', verbose_name="错误信息")
    should_stop = models.BooleanField(default=False, verbose_name="停止标志")

    # 时间字段
    training_start_time = models.DateTimeField(null=True, blank=True, verbose_name="训练开始时间")
    training_end_time = models.DateTimeField(null=True, blank=True, verbose_name="训练结束时间")

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "训练状态"
        verbose_name_plural = "训练状态"
        db_table = 'training_status'

    def __str__(self):
        return f"训练状态 - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"

    def to_dict(self):
        return {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'eval_f1': self.eval_f1,
            'eval_accuracy': self.eval_accuracy,
            'eval_recall': self.eval_recall,
            'progress_percent': self.progress_percent,
            'logs': self.logs,
            'error_message': self.error_message,
            'final_metrics': self.final_metrics,
            'hyperparameters': self.hyperparameters,
            'training_start_time': self.training_start_time,
            'training_end_time': self.training_end_time,
            'should_stop': self.should_stop,
        }

    @classmethod
    def from_dict(cls, data):
        """从字典创建实例"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
