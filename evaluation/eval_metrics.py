"""
TRUS报告生成评估：NLG指标 + 结构化字段准确率 + 诊断分类指标
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TEST_JSON_PATH, PROCESSED_DIR, OUTPUT_DIR


def compute_bleu(reference: str, candidate: str, max_n: int = 4) -> Dict[str, float]:
    """计算 BLEU-1 到 BLEU-N，使用 nltk 实现。"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    ref_tokens = [list(reference)]
    cand_tokens = list(candidate)
    scores = {}
    for n in range(1, max_n + 1):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        try:
            score = sentence_bleu(ref_tokens, cand_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f"bleu_{n}"] = score * 100
    return scores


def compute_rouge_l(reference: str, candidate: str) -> float:
    """计算 ROUGE-L F1。"""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, candidate)
    return result["rougeL"].fmeasure * 100


def compute_meteor(reference: str, candidate: str) -> float:
    """计算 METEOR 分数。"""
    from nltk.translate.meteor_score import meteor_score
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    return meteor_score([reference.split()], candidate.split()) * 100


def compute_bert_score(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """计算 BERTScore (Precision/Recall/F1)。"""
    from bert_score import score
    P, R, F1 = score(candidates, references, lang="zh", verbose=False)
    return {
        "bert_precision": P.mean().item() * 100,
        "bert_recall": R.mean().item() * 100,
        "bert_f1": F1.mean().item() * 100,
    }


def evaluate_nlg(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """综合 NLG 评估。"""
    results = {}
    bleu_scores = defaultdict(list)
    rouge_scores = []
    meteor_scores = []

    for ref, pred in zip(references, predictions):
        for k, v in compute_bleu(ref, pred).items():
            bleu_scores[k].append(v)
        rouge_scores.append(compute_rouge_l(ref, pred))
        try:
            meteor_scores.append(compute_meteor(ref, pred))
        except Exception:
            pass

    for k, v in bleu_scores.items():
        results[k] = np.mean(v)
    results["rouge_l"] = np.mean(rouge_scores)
    results["meteor"] = np.mean(meteor_scores)
    results.update(compute_bert_score(references, predictions))
    return results


def evaluate_structured_fields(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """评估结构化字段准确率：尺寸、形态、回声、钙化、血流。"""
    from data.report_parser import parse_trus_report

    field_correct = defaultdict(int)
    field_total = 0
    dim_errors = {"lr": [], "ap": [], "si": []}

    for pred, ref in zip(predictions, references):
        pred_fields = parse_trus_report(pred)
        ref_fields = parse_trus_report(ref)
        field_total += 1

        # 形态
        if pred_fields.get("shape") == ref_fields.get("shape") and ref_fields.get("shape") is not None:
            field_correct["shape"] += 1
        # 回声
        if pred_fields.get("echotexture") == ref_fields.get("echotexture") and ref_fields.get("echotexture") is not None:
            field_correct["echotexture"] += 1
        # 钙化
        if pred_fields.get("calcification") == ref_fields.get("calcification") and ref_fields.get("calcification") is not None:
            field_correct["calcification"] += 1
        # 血流
        if pred_fields.get("blood_flow") == ref_fields.get("blood_flow") and ref_fields.get("blood_flow") is not None:
            field_correct["blood_flow"] += 1
        # 包膜
        if pred_fields.get("capsule") == ref_fields.get("capsule") and ref_fields.get("capsule") is not None:
            field_correct["capsule"] += 1
        # 尺寸MAE
        for dim_key in ["lr_diameter", "ap_diameter", "si_diameter"]:
            pv = pred_fields.get("dimensions", {}).get(dim_key)
            rv = ref_fields.get("dimensions", {}).get(dim_key)
            if pv is not None and rv is not None:
                dim_errors[dim_key.split("_")[0]].append(abs(pv - rv))

    results = {}
    for field in ["shape", "echotexture", "calcification", "blood_flow", "capsule"]:
        results[f"acc_{field}"] = field_correct[field] / field_total * 100 if field_total > 0 else 0
    for k, v in dim_errors.items():
        results[f"mae_{k}_mm"] = np.mean(v) if v else float("nan")

    return results


def evaluate_classification(predictions: List[str], labels: List[bool]) -> Dict[str, float]:
    """从生成的报告文本中判断良恶性，与金标准比对。"""
    cancer_keywords = ["癌", "恶性", "占位", "结节待查", "可疑", "异常回声区", "低回声结节"]
    pred_labels = []
    for pred in predictions:
        has_cancer = any(kw in pred for kw in cancer_keywords)
        pred_labels.append(has_cancer)

    y_true = np.array(labels, dtype=int)
    y_pred = np.array(pred_labels, dtype=int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results = {
        "cls_accuracy": acc * 100,
        "cls_precision": prec * 100,
        "cls_recall": rec * 100,
        "cls_f1": f1 * 100,
        "cls_auc": auc * 100,
        "cls_sensitivity": tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        "cls_specificity": tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
    }
    return results


def run_full_evaluation(predictions_json: Optional[Path] = None):
    """运行完整评估管线。"""
    if predictions_json is None:
        predictions_json = OUTPUT_DIR / "test_predictions.json"

    if not predictions_json.exists():
        print(f"[错误] 预测文件不存在: {predictions_json}")
        print("请先运行推理脚本生成预测结果。")
        return

    with open(predictions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = [item["prediction"] for item in data]
    references = [item["reference"] for item in data]
    labels = [item.get("metadata", {}).get("has_cancer", False) for item in data]

    print("=" * 60)
    print("  NLG 评估结果")
    print("=" * 60)
    nlg_results = evaluate_nlg(predictions, references)
    for k, v in nlg_results.items():
        print(f"  {k:20s}: {v:.2f}")

    print("\n" + "=" * 60)
    print("  结构化字段评估结果")
    print("=" * 60)
    struct_results = evaluate_structured_fields(predictions, references)
    for k, v in struct_results.items():
        print(f"  {k:20s}: {v:.2f}")

    if any(l is not None for l in labels):
        print("\n" + "=" * 60)
        print("  癌症诊断分类评估结果")
        print("=" * 60)
        cls_results = evaluate_classification(predictions, labels)
        for k, v in cls_results.items():
            print(f"  {k:20s}: {v:.2f}")

    all_results = {**nlg_results, **struct_results}
    save_path = OUTPUT_DIR / "evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存至: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default=None, help="预测JSON路径")
    args = parser.parse_args()
    run_full_evaluation(Path(args.predictions) if args.predictions else None)
