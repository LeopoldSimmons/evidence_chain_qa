# -*- coding: utf-8 -*-
"""评估模块：计算证据检索与答案生成的基础指标。"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

from .qa_system import EvidenceChainQA


def f1_score(pred: Set[int], gold: Set[int]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[\s，。！？；：、,.!?;:()（）\[\]{}<>《》\"'“”‘’]+", "", text)
    return text


def char_f1(pred: str, gold: str) -> float:
    pred = normalize_answer(pred)
    gold = normalize_answer(gold)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pc = Counter(pred)
    gc = Counter(gold)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def evaluate(system: EvidenceChainQA, qa_path: str) -> Dict:
    gold_items: List[Dict] = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    rows = []
    ev_f1_list = []
    answer_f1_list = []
    answer_match_list = []
    for item in gold_items:
        pred = system.answer(item["question"])
        pred_set = set(pred["evidence_ids"])
        gold_set = set(item["evidence"])
        ev_f1 = f1_score(pred_set, gold_set)
        ans_f1 = char_f1(pred["answer"], item["answer"])
        # 复杂解释题很难逐字完全一致，因此采用字符级F1阈值作为近似答案匹配。
        answer_match = 1.0 if ans_f1 >= 0.55 else 0.0
        rows.append({
            "question": item["question"],
            "gold_answer": item["answer"],
            "pred_answer": pred["answer"],
            "gold_evidence": item["evidence"],
            "pred_evidence": pred["evidence_ids"],
            "evidence_f1": round(ev_f1, 4),
            "answer_char_f1": round(ans_f1, 4),
            "answer_match": answer_match,
        })
        ev_f1_list.append(ev_f1)
        answer_f1_list.append(ans_f1)
        answer_match_list.append(answer_match)
    return {
        "num_questions": len(gold_items),
        "mean_evidence_f1": round(sum(ev_f1_list) / max(1, len(ev_f1_list)), 4),
        "mean_answer_char_f1": round(sum(answer_f1_list) / max(1, len(answer_f1_list)), 4),
        "answer_match_accuracy": round(sum(answer_match_list) / max(1, len(answer_match_list)), 4),
        "details": rows,
    }
