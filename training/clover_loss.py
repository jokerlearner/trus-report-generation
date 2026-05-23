"""
CLOVER 多目标损失函数。
L = L_LM + λ₁·L_CLS + λ₂·L_CONTRAST

MoCo 动量队列设计:
- 每个 micro-batch 的 (visual_emb, clinical_emb) 以 detach 方式入队
- 当前 micro-batch 的 visual_emb 作为 query，队列中的 clinical_emb 作为 keys
- 队列提供足够的负样本，即使 batch_size=1 也能计算有意义的 InfoNCE
- 队列按先进先出更新，大小由 moco_queue_size 控制
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLOVERLoss:
    """
    ms-swift loss_map 兼容的损失函数。

    ms-swift BaseLoss 协议:
        __call__(self, outputs, labels, *, num_items_in_batch, loss_scale, **kwargs)
    kwargs 中包含 trainer=self (由 ms-swift Seq2SeqTrainer.compute_loss 传入)。
    """

    def __init__(self, args=None, trainer=None):
        self.args = args
        self.trainer = trainer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self._temperature = 0.5  # scalar float, updated only at step boundaries
        self._temperature_tensor = torch.tensor(0.5)  # GPU tensor copy for computation

        # MoCo 队列 (在首次调用时按实际 device 初始化)
        self.queue_visual = None       # [queue_size, 256]
        self.queue_clinical = None     # [queue_size, 256]
        self.queue_ptr = 0
        self.queue_full = False
        self._queue_initialized = False

    def _ensure_queue(self, device, queue_size: int = 32, proj_dim: int = 256):
        if not self._queue_initialized:
            self.queue_visual = torch.zeros(queue_size, proj_dim, device=device)
            self.queue_clinical = torch.zeros(queue_size, proj_dim, device=device)
            self.queue_ptr = 0
            self.queue_full = False
            self._queue_initialized = True

    @torch.no_grad()
    def _dequeue_and_enqueue(self, visual: torch.Tensor, clinical: torch.Tensor):
        """将当前 batch 的 detach 后的 embedding 加入 MoCo 队列 (FIFO)。
        使用 tensor 替换而非 in-place 修改，避免 autograd 版本冲突。"""
        batch_size = visual.shape[0]
        queue_size = self.queue_visual.shape[0]
        ptr = self.queue_ptr

        # 创建新的 queue tensor (避免 in-place 修改)
        new_visual = self.queue_visual.clone()
        new_clinical = self.queue_clinical.clone()

        if ptr + batch_size <= queue_size:
            new_visual[ptr:ptr + batch_size] = visual
            new_clinical[ptr:ptr + batch_size] = clinical
        else:
            space = queue_size - ptr
            new_visual[ptr:] = visual[:space]
            new_clinical[ptr:] = clinical[:space]
            remaining = batch_size - space
            if remaining > 0:
                new_visual[:remaining] = visual[space:]
                new_clinical[:remaining] = clinical[space:]

        self.queue_visual = new_visual
        self.queue_clinical = new_clinical

        self.queue_ptr = (ptr + batch_size) % queue_size
        if not self.queue_full and self.queue_ptr == 0:
            self.queue_full = True

    # ======================== 主入口 ========================

    def __call__(self, outputs, labels, *, num_items_in_batch=None,
                 loss_scale=None, trainer=None, **kwargs):
        model = trainer.model
        if hasattr(model, "module"):
            model = model.module

        device = outputs.logits.device
        batch_size = outputs.logits.shape[0] if outputs.logits is not None else 1

        # 初始化 MoCo 队列
        queue_size = getattr(trainer.args, "_clover_queue_size", 32)
        self._ensure_queue(device, queue_size=queue_size, proj_dim=256)

        # 仅在梯度累积边界同步温度 tensor (避免计算图版本冲突)
        self._sync_temperature(device)

        # ---- 1. 语言建模损失 (从logits+labels自行计算，不依赖outputs.loss) ----
        if labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            lm_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # ---- 2. 癌症分类损失 (BCE + Label Smoothing) ----
        cls_loss = self._compute_cls_loss(outputs, trainer, model, device)

        # ---- 3. 临床-视觉对比对齐损失 (MoCo InfoNCE) ----
        contrast_loss = self._compute_contrastive_loss(outputs, trainer, model, device)

        # ---- 获取 warmup 后的 λ 权重 ----
        lambda1, lambda2 = self._get_lambda_values(trainer)

        # ---- 温度退火 ----
        current_step = trainer.state.global_step if trainer.state else 0
        max_steps = trainer.state.max_steps if trainer.state and trainer.state.max_steps else 336
        self._anneal_temperature(current_step, max_steps)

        # ---- 组合损失 ----
        total_loss = lm_loss + lambda1 * cls_loss + lambda2 * contrast_loss

        # ---- 记录各分量 ----
        if hasattr(trainer, "log") and trainer.is_world_process_zero():
            # 首次调试：确认LM loss非零
            if not hasattr(self, "_debug_printed"):
                self._debug_printed = True
                non_masked = (labels != -100).sum().item() if labels is not None else 0
                print(f"[CLOVER DEBUG] lm_loss={lm_loss.item():.4f}, "
                      f"cls_loss={cls_loss.item():.4f}, "
                      f"labels_non_masked={non_masked}, "
                      f"total_labels={labels.numel() if labels is not None else 0}", flush=True)
            trainer.log({
                "loss/lm": lm_loss.item() if hasattr(lm_loss, 'item') else float(lm_loss),
                "loss/cls": cls_loss.item(),
                "loss/contrast": contrast_loss.item(),
                "loss/lambda1": lambda1,
                "loss/lambda2": lambda2,
                "loss/temperature": self._temperature,
                "loss/moco_queue_full": 1.0 if self.queue_full else 0.0,
            })

        if loss_scale is not None:
            total_loss = total_loss * loss_scale

        return total_loss

    # ======================== 分类损失 ========================

    def _compute_cls_loss(self, outputs, trainer, model, device) -> torch.Tensor:
        """BCE 分类损失，含标签平滑。"""
        clinical = trainer._clover_batch_cache if hasattr(trainer, "_clover_batch_cache") else {}
        has_cancer = clinical.get("has_cancer")
        if has_cancer is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        h_last = self._get_last_hidden(outputs)
        if h_last is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits = model.cancer_classifier(h_last).squeeze(-1)

        # Label smoothing (从config读取)
        labels = has_cancer.float()
        eps = getattr(trainer.args, "_clover_label_smoothing", 0.1)
        labels_smoothed = labels * (1.0 - eps) + 0.5 * eps

        return F.binary_cross_entropy_with_logits(logits, labels_smoothed)

    # ======================== 对比损失 (MoCo InfoNCE) ========================

    def _compute_contrastive_loss(self, outputs, trainer, model, device) -> torch.Tensor:
        """MoCo InfoNCE: visual query 检索 clinical keys 队列。"""
        clinical = trainer._clover_batch_cache if hasattr(trainer, "_clover_batch_cache") else {}
        psa = clinical.get("psa")
        volume = clinical.get("volume")
        has_cancer = clinical.get("has_cancer")

        if psa is None or volume is None or has_cancer is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        h_last = self._get_last_hidden(outputs)
        if h_last is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 临床特征向量 [B, 3] — z-score 标准化
        psa_norm = self._normalize(psa, trainer, "psa")
        vol_norm = self._normalize(volume, trainer, "volume")
        clinical_feat = torch.stack([psa_norm, vol_norm, has_cancer.float()], dim=1)

        # 随机遮蔽临床特征 (防过拟合: 迫使模型不完全依赖临床参数)
        mask_prob = getattr(trainer.args, "_clover_feature_mask_prob", 0.2)
        if mask_prob > 0:
            mask = torch.bernoulli(torch.full_like(clinical_feat, 1.0 - mask_prob))
            clinical_feat = clinical_feat * mask

        # 临床 embedding [B, 256]
        clinical_emb = model.clinical_mlp(clinical_feat)
        clinical_emb = F.normalize(clinical_emb, dim=1)

        # 视觉 embedding [B, 256]
        visual_emb = model.visual_projection(h_last)
        visual_emb = F.normalize(visual_emb, dim=1)

        # ---- MoCo InfoNCE ----
        if self.queue_full:
            # Positive: visual query × clinical key (同一患者, 对角线)
            l_pos = torch.sum(visual_emb * clinical_emb, dim=1, keepdim=True)  # [B, 1]
            # Negative: visual query × queue clinical keys
            l_neg = torch.matmul(visual_emb, self.queue_clinical.T)  # [B, Q]
            logits = torch.cat([l_pos, l_neg], dim=1) / self._temperature_tensor  # [B, 1+Q]
            labels = torch.zeros(visual_emb.shape[0], dtype=torch.long, device=device)
            contrast_loss = F.cross_entropy(logits, labels)
        else:
            # 队列未满时仅用批内样本做对比 (可能退化, 但 λ₂ 也还在 warmup 中)
            sim = torch.matmul(visual_emb, clinical_emb.T) / self._temperature_tensor
            labels = torch.arange(visual_emb.shape[0], device=device)
            contrast_loss = F.cross_entropy(sim, labels)

        # 当前 batch 入队 (detach)
        self._dequeue_and_enqueue(visual_emb.detach(), clinical_emb.detach())

        return contrast_loss

    # ======================== 辅助方法 ========================

    def _get_last_hidden(self, outputs):
        """从 outputs 中提取最后一层最后 token 的 hidden state。"""
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1][:, -1, :]
        # fallback: 尝试 last_hidden_state (某些自定义 forward 可能返回)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return None

    def _normalize(self, tensor: torch.Tensor, trainer, key: str) -> torch.Tensor:
        """Z-score 标准化。统计量预先计算并存储在 trainer 上。"""
        mean = getattr(trainer, f"_clover_{key}_mean", 0.0)
        std = getattr(trainer, f"_clover_{key}_std", 1.0)
        # 处理 NaN (缺失值) → 用均值填充
        tensor = torch.where(tensor.isnan(), torch.tensor(mean, device=tensor.device), tensor)
        return (tensor.float() - mean) / (std + 1e-8)

    def _get_lambda_values(self, trainer):
        """
        λ 渐进 warmup:
        λ₁ (cls): 0 → lambda_cls (0.3) over cls_warmup_steps
        λ₂ (contrast): 0 → lambda_contrast (0.15) over contrast_warmup_steps
        """
        step = trainer.state.global_step if trainer.state else 0
        lambda_cls = getattr(trainer.args, "_clover_lambda_cls", 0.3)
        lambda_contrast = getattr(trainer.args, "_clover_lambda_contrast", 0.15)
        warmup_cls = getattr(trainer.args, "_clover_cls_warmup", 50)
        warmup_contrast = getattr(trainer.args, "_clover_contrast_warmup", 100)

        l1 = min(1.0, step / max(warmup_cls, 1)) * lambda_cls
        l2 = min(1.0, step / max(warmup_contrast, 1)) * lambda_contrast
        return l1, l2

    def _sync_temperature(self, device):
        """同步温度 scalar → GPU tensor (仅在需要时创建新 tensor, 避免在梯度累积中替换)。"""
        target = self._temperature
        if self._temperature_tensor.device != device:
            self._temperature_tensor = torch.tensor(target, device=device)
        elif abs(self._temperature_tensor.item() - target) > 1e-8:
            self._temperature_tensor = torch.tensor(target, device=device)

    def _anneal_temperature(self, current_step: int, max_steps: int):
        """Cosine 退火: 0.5 → 0.1。仅在 step 边界更新,不创建新 tensor。"""
        tau_init = 0.5
        tau_min = 0.1
        progress = min(current_step / max(max_steps, 1), 1.0)
        self._temperature = tau_min + 0.5 * (tau_init - tau_min) * (1.0 + math.cos(math.pi * progress))
