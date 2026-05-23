"""
CLOVER 外部插件入口 — ms-swift external_plugins 机制。
通过 SftArguments(external_plugins=["training/clover_plugin.py"]) 加载。

该脚本在 ms-swift 初始化阶段导入，完成:
1. 注册自定义损失函数 (loss_map["clover"] = CLOVERLoss)
2. 在 Trainer 创建后注入 CLOVER heads (cancer_classifier / visual_projection / clinical_mlp)
3. 在训练步中提取临床特征 (has_cancer / psa / volume) 供 loss 函数使用
4. 计算并存储临床特征 z-score 标准化统计量
"""

import torch
import torch.nn as nn

from training.clover_loss import CLOVERLoss
from training.clover_components import ClinicalMLP


# ======================== 1. 注册损失函数 ========================

def _register_loss():
    try:
        from swift.loss.mapping import loss_map
        loss_map["clover"] = CLOVERLoss
        print("[CLOVER] 损失函数已注册: loss_type='clover' -> CLOVERLoss")
    except Exception as e:
        print(f"[CLOVER] 警告: 无法注册损失函数: {e}")


_register_loss()


# ======================== 2. 注入 CLOVER heads 到模型 ========================

def _ensure_clover_heads(trainer):
    """惰性初始化: 在首次训练步时将 CLOVER heads 附加到模型上。"""
    model = trainer.model
    if hasattr(model, "module"):
        model = model.module

    # 避免重复附加
    if hasattr(model, "cancer_classifier") and hasattr(model, "visual_projection"):
        return

    hidden_size = model.config.hidden_size  # 3584

    # B'': 癌症分类头 (3584 → 1)
    model.cancer_classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(hidden_size, 1),
    ).to(model.device)

    # D: 视觉投影头 (3584 → 256 → 256) — 用于对比学习
    model.visual_projection = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 256),
    ).to(model.device)

    # D: 临床特征编码器 (3 → 64 → 256)
    model.clinical_mlp = ClinicalMLP(
        input_dim=3, hidden_dim=64, output_dim=256, dropout=0.2
    ).to(model.device)

    # Xavier 初始化
    for head in [model.cancer_classifier, model.visual_projection, model.clinical_mlp]:
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # 确保新 heads 可训练
    for head in [model.cancer_classifier, model.visual_projection, model.clinical_mlp]:
        for p in head.parameters():
            p.requires_grad = True

    print(f"[CLOVER] Heads 已注入: cancer_classifier + visual_projection + clinical_mlp (device={model.device})")


# ======================== 3. 训练步注入 ========================

def _cache_clinical_features(trainer, inputs):
    """从 batch inputs 中提取临床特征，缓存到 trainer 上供 loss 函数访问。"""
    device = None
    model = trainer.model
    if hasattr(model, "module"):
        model = model.module
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = {}
    for key in ["has_cancer", "psa", "volume"]:
        if key in inputs:
            val = inputs[key]
            if isinstance(val, torch.Tensor):
                features[key] = val.to(device)
            elif isinstance(val, list):
                tensor_vals = []
                for v in val:
                    if v is None:
                        tensor_vals.append(float("nan"))
                    elif isinstance(v, bool):
                        tensor_vals.append(1.0 if v else 0.0)
                    else:
                        try:
                            tensor_vals.append(float(v))
                        except (ValueError, TypeError):
                            tensor_vals.append(float("nan"))
                features[key] = torch.tensor(tensor_vals, device=device, dtype=torch.float32)

    trainer._clover_batch_cache = features


def _patch_training_step():
    """Monkey-patch: 在训练步中注入 CLOVER 逻辑。"""
    try:
        from swift.trainers import Seq2SeqTrainer
    except ImportError:
        try:
            from swift.trainers.trainers import Seq2SeqTrainer
        except ImportError:
            print("[CLOVER] 警告: 无法导入 Seq2SeqTrainer, 跳过 training_step patch")
            return

    original_training_step = Seq2SeqTrainer.training_step

    def clover_training_step(self, model, inputs, *args, **kwargs):
        _ensure_clover_heads(self)
        _cache_clinical_features(self, inputs)

        # 确保 hidden_states 被返回
        if "output_hidden_states" not in inputs:
            inputs = dict(inputs)
        inputs["output_hidden_states"] = True

        return original_training_step(self, model, inputs, *args, **kwargs)

    Seq2SeqTrainer.training_step = clover_training_step
    print("[CLOVER] training_step patch 已应用")


_patch_training_step()


# ======================== 4. 临床特征统计量 (z-score) ========================

def _compute_clinical_stats_remote(trainer):
    """从训练集中计算 PSA/Volume 的均值和标准差，供 z-score 标准化使用。"""
    try:
        train_dataset = trainer.train_dataset
        psa_vals, vol_vals = [], []
        for sample in train_dataset:
            psa = sample.get("psa")
            volume = sample.get("volume")
            if psa is not None:
                try:
                    psa_vals.append(float(psa))
                except (ValueError, TypeError):
                    pass
            if volume is not None:
                try:
                    vol_vals.append(float(volume))
                except (ValueError, TypeError):
                    pass

        if psa_vals:
            psa_mean = sum(psa_vals) / len(psa_vals)
            psa_var = sum((x - psa_mean) ** 2 for x in psa_vals) / len(psa_vals)
            psa_std = psa_var ** 0.5
        else:
            psa_mean, psa_std = 15.0, 20.0

        if vol_vals:
            vol_mean = sum(vol_vals) / len(vol_vals)
            vol_var = sum((x - vol_mean) ** 2 for x in vol_vals) / len(vol_vals)
            vol_std = vol_var ** 0.5
        else:
            vol_mean, vol_std = 50.0, 30.0

        trainer._clover_psa_mean = psa_mean
        trainer._clover_psa_std = max(psa_std, 1e-6)
        trainer._clover_volume_mean = vol_mean
        trainer._clover_volume_std = max(vol_std, 1e-6)

        print(f"[CLOVER] 临床统计: PSA mean={psa_mean:.1f} std={psa_std:.1f}, "
              f"Volume mean={vol_mean:.1f} std={vol_std:.1f}")
    except Exception as e:
        print(f"[CLOVER] 警告: 无法计算临床统计量: {e}")
        trainer._clover_psa_mean = 15.0
        trainer._clover_psa_std = 20.0
        trainer._clover_volume_mean = 50.0
        trainer._clover_volume_std = 30.0


# 在训练开始时计算统计量: 通过 patch Trainer.train() 实现
def _patch_train_for_stats():
    try:
        from swift.trainers import Seq2SeqTrainer
    except ImportError:
        try:
            from swift.trainers.trainers import Seq2SeqTrainer
        except ImportError:
            return

    original_train = Seq2SeqTrainer.train

    def clover_train(self, *args, **kwargs):
        # 如果尚未初始化 heads (可能在 resume 场景)
        _ensure_clover_heads(self)
        # 计算临床统计量 (仅 rank 0 打印)
        if not hasattr(self, "_clover_psa_mean"):
            _compute_clinical_stats_remote(self)
        return original_train(self, *args, **kwargs)

    Seq2SeqTrainer.train = clover_train
    print("[CLOVER] train() patch 已应用 (临床统计量自动计算)")


_patch_train_for_stats()

# ======================== 5. 传递 CLOVER 配置到 trainer.args ========================

def _inject_clover_config():
    """从 config.py 的 CLOVER_CONFIG 读取配置并注入到训练参数中。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from config import CLOVER_CONFIG
    except ImportError:
        print("[CLOVER] 警告: 无法导入 CLOVER_CONFIG, 使用默认值")
        return

    # 延迟注入: 在 SftArguments 构建后通过 monkey-patch __post_init__ 实现
    # 这里用更简单的方式: 在训练开始时通过 trainer.args 注入
    def _patch_args():
        try:
            from swift.trainers import Seq2SeqTrainer
        except ImportError:
            try:
                from swift.trainers.trainers import Seq2SeqTrainer
            except ImportError:
                return

        original_init = Seq2SeqTrainer.__init__

        def clover_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # 将 CLOVER 配置附加到 args 上
            for key, val in CLOVER_CONFIG.items():
                setattr(self.args, f"_clover_{key}", val)

        Seq2SeqTrainer.__init__ = clover_init
        print("[CLOVER] Trainer.__init__ patch 已应用 (CLOVER_CONFIG 注入)")

    _patch_args()


_inject_clover_config()

print("[CLOVER] 插件加载完成")
