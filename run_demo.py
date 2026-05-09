# -*- coding: utf-8 -*-
"""命令行演示入口。运行：python run_demo.py"""
import json
from pathlib import Path

from src.evaluator import evaluate
from src.qa_system import EvidenceChainQA, pretty_print_result

BASE_DIR = Path(__file__).resolve().parent
COMMENTARY_PATH = BASE_DIR / "data" / "commentary_sample.json"
QA_GOLD_PATH = BASE_DIR / "data" / "qa_gold.json"
OUTPUT_DIR = BASE_DIR / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    system = EvidenceChainQA(str(COMMENTARY_PATH))

    # 选取多跳问题进行演示，避免只依靠比分或单个关键词即可猜中。
    demo_questions = [
        "拜仁第三个进球前，同一分钟出现了哪次更早的关键机会？这两条证据连起来说明了什么？",
        "Michael Gregoritsch这名球员的事件链有什么特点？请结合他早期参与进攻、犯规、吃牌和被换下的证据回答。",
        "弗赖堡在下半场刚开始是否曾尝试反扑？请找出46到47分钟之间能支持这一判断的连续证据。",
        "弗赖堡为什么可以被认为在50到51分钟出现了短时间纪律崩盘？",
        "拜仁的第五个进球和76分钟的换人调整之间有什么直接关系？",
    ]

    all_results = []
    print("=" * 80)
    print("证据链检索增强问答系统 Demo：拜仁慕尼黑 5-0 弗赖堡")
    print("=" * 80)
    for q in demo_questions:
        result = system.answer(q)
        all_results.append(result)
        pretty_print_result(result)
        print("-" * 80)

    eval_result = evaluate(system, str(QA_GOLD_PATH))
    print("评估结果:")
    print(json.dumps(eval_result, ensure_ascii=False, indent=2))

    (OUTPUT_DIR / "demo_predictions.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "evaluation.json").write_text(
        json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
