# -*- coding: utf-8 -*-
"""生成课程报告中使用的补充统计图表与消融实验结果。"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25
from src.qa_system import EvidenceChainQA, EVENT_HINTS
from src.text_utils import char_ngrams, expand_query
from src.evaluator import f1_score

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

commentary = json.loads((DATA / 'commentary_sample.json').read_text(encoding='utf-8'))
qa_items = json.loads((DATA / 'qa_gold.json').read_text(encoding='utf-8'))

# 1. 事件类型分布
etype_counter = Counter(e.get('type', 'unknown') for e in commentary)
major = etype_counter.most_common(12)
plt.figure(figsize=(10, 5))
plt.bar([x[0] for x in major], [x[1] for x in major])
plt.title('赛事评论事件类型分布')
plt.xlabel('事件类型')
plt.ylabel('事件数量')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig(OUT / 'event_type_distribution.png', dpi=180)
plt.close()

# 2. 问题类型分布
QTYPE_ZH = {
    'temporal_multi_hop': '多时间跳跃',
    'player_event_chain': '球员事件链',
    'temporal_window_reasoning': '时间窗口推理',
    'event_cluster_reasoning': '事件聚集推理',
    'substitution_to_goal': '换人-进球关系',
    'player_progression': '球员进攻轨迹',
    'role_change_reasoning': '角色变化推理',
    'set_reasoning': '集合统计推理',
    'causal_context': '背景因果推理',
    'comparative_reasoning': '对比推理',
    'long_range_evidence': '长距离证据',
}
qtype_counter_raw = Counter(item.get('question_type', 'unknown') for item in qa_items)
qtype_counter = Counter({QTYPE_ZH.get(k, k): v for k, v in qtype_counter_raw.items()})
items = sorted(qtype_counter.items(), key=lambda x: x[1])
plt.figure(figsize=(7.5, 5.2))
plt.barh([x[0] for x in items], [x[1] for x in items])
plt.title('复杂问答样例类型分布')
plt.xlabel('问题数量')
plt.ylabel('问题类型')
plt.tight_layout()
plt.savefig(OUT / 'qa_type_distribution.png', dpi=180)
plt.close()

# 3. 证据链长度分布
lengths = [len(item.get('evidence', [])) for item in qa_items]
len_counter = Counter(lengths)
plt.figure(figsize=(7, 4.5))
plt.bar([str(k) for k in sorted(len_counter)], [len_counter[k] for k in sorted(len_counter)])
plt.title('问答样例证据链长度分布')
plt.xlabel('证据事件数量')
plt.ylabel('问题数量')
plt.tight_layout()
plt.savefig(OUT / 'evidence_length_distribution.png', dpi=180)
plt.close()

# 4. 消融实验：比较纯 BM25、BM25+领域加权、完整证据链规则
system = EvidenceChainQA(str(DATA / 'commentary_sample.json'))
docs = [system._event_to_document(e) for e in system.events]
doc_tokens = [char_ngrams(doc) for doc in docs]
bm25 = BM25(doc_tokens)

# helper: 消融实验统一采用固定 top_k=6，而不是按人工证据数量取 k。
# 这样可以同时观察候选证据的冗余程度和漏检程度，平均准确率与平均召回率不再必然相同。
def bm25_only(question, k=6):
    q_tokens = char_ngrams(question)
    idxs = [idx for idx, _ in bm25.rank(q_tokens, top_k=k)]
    return [system.events[i]['event_id'] for i in idxs]

def bm25_event_entity_filter(question, k=6):
    target_types = set(system.infer_target_event_types(question))
    target_team = system.extract_target_team(question)
    ranked_events = [system.events[idx] for idx, _ in bm25.rank(char_ngrams(question), top_k=len(system.events))]
    filtered = []
    for e in ranked_events:
        ok = True
        if target_types and e.get('type') not in target_types:
            ok = False
        if target_team and e.get('team') != target_team:
            ok = False
        if ok:
            filtered.append(e)
    seen = {e['event_id'] for e in filtered}
    for e in ranked_events:
        if len(filtered) >= k:
            break
        if e['event_id'] not in seen:
            filtered.append(e)
            seen.add(e['event_id'])
    return [e['event_id'] for e in filtered[:k]]

def full_chain(question, k=6):
    return system.answer(question)['evidence_ids'][:k]

variants = {
    '纯BM25检索': bm25_only,
    'BM25+事件/实体过滤': bm25_event_entity_filter,
    '完整证据链系统': full_chain,
}
abla = []
FIXED_TOP_K = 6
for name, fn in variants.items():
    f1s, ps, rs = [], [], []
    for item in qa_items:
        gold = set(item['evidence'])
        pred = set(fn(item['question'], FIXED_TOP_K))
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold) if gold else 0.0
        f = f1_score(pred, gold)
        ps.append(p); rs.append(r); f1s.append(f)
    abla.append({
        'variant': name,
        'precision': round(sum(ps)/len(ps), 4),
        'recall': round(sum(rs)/len(rs), 4),
        'evidence_f1': round(sum(f1s)/len(f1s), 4),
    })

with open(OUT / 'ablation_results.json', 'w', encoding='utf-8') as f:
    json.dump(abla, f, ensure_ascii=False, indent=2)

plt.figure(figsize=(8, 4.8))
labels = [x['variant'] for x in abla]
vals = [x['evidence_f1'] for x in abla]
plt.bar(labels, vals)
plt.title('不同检索策略的平均证据F1对比')
plt.xlabel('系统变体')
plt.ylabel('平均证据F1')
plt.ylim(0, 1.05)
for i, v in enumerate(vals):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.savefig(OUT / 'ablation_comparison.png', dpi=180)
plt.close()

# 5. 汇总统计
team_counter = Counter(e.get('team') or '无明确球队' for e in commentary)
minute_counter = defaultdict(int)
for e in commentary:
    t = e.get('time', '')
    if isinstance(t, str) and ':' in t:
        minute = int(t.split(':')[0])
        half = '上半场' if minute < 45 else '下半场'
    else:
        half = '无时间'
    minute_counter[half] += 1

stats = {
    'num_events': len(commentary),
    'num_qa': len(qa_items),
    'event_type_counter': dict(etype_counter),
    'team_counter': dict(team_counter),
    'half_counter': dict(minute_counter),
    'question_type_counter': dict(qtype_counter),
    'question_type_counter_raw': dict(qtype_counter_raw),
    'evidence_length_counter': {str(k): v for k, v in sorted(len_counter.items())},
    'ablation': abla,
}
with open(OUT / 'extra_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print('已生成补充图表与统计结果：')
for p in ['event_type_distribution.png', 'qa_type_distribution.png', 'evidence_length_distribution.png', 'ablation_comparison.png', 'extra_stats.json', 'ablation_results.json']:
    print(OUT / p)
