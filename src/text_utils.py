# -*- coding: utf-8 -*-
"""文本预处理与轻量中文切分工具。"""
import re
from typing import List

# 课程作业原型尽量减少依赖，因此采用字符n-gram而不是外部分词模型。
# 同义词扩展用于缓解体育评论中“进球/破门”“黄牌/吃牌”等表达差异。
SYNONYM_MAP = {
    "进球": ["破门", "得分", "打进", "比分"],
    "破门": ["进球", "得分"],
    "黄牌": ["吃牌", "出示黄牌"],
    "犯规": ["撞倒", "拉拽", "战术犯规"],
    "换人": ["换下", "换上", "替补", "登场", "调整", "replaces"],
    "反超": ["领先", "逆转", "扳平"],
    "射门": ["远射", "头球", "低射", "被封堵", "被扑出", "门柱"],
    "角球": ["定位球", "corner"],
    "越位": ["offside"],
    "助攻": ["assisted", "assisted by"],
    "弗赖堡": ["Freiburg", "SC Freiburg"],
    "拜仁": ["Bayern", "FC Bayern", "拜仁慕尼黑"],
    "时间": ["什么时候", "发生在"],
}

PUNCT_RE = re.compile(r"[\s，。！？；：、,.!?;:()（）\[\]{}<>《》\"'“”‘’]+")


def normalize_text(text: str) -> str:
    """统一大小写、空白和常见标点。"""
    text = text.lower().strip()
    return PUNCT_RE.sub("", text)


def char_ngrams(text: str, min_n: int = 1, max_n: int = 3) -> List[str]:
    """生成字符级n-gram，适合短中文事件句检索。"""
    text = normalize_text(text)
    grams: List[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(max(0, len(text) - n + 1)):
            grams.append(text[i:i + n])
    return grams


def expand_query(query: str) -> str:
    """基于小型同义词表扩展查询，提升召回率。"""
    expanded = query
    for key, synonyms in SYNONYM_MAP.items():
        if key in query:
            expanded += " " + " ".join(synonyms)
    return expanded


def extract_number(text: str):
    """抽取阿拉伯数字或中文数字，返回整数；无法识别时返回None。"""
    m = re.search(r"\d+", text)
    if m:
        return int(m.group())
    zh_map = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    for k, v in zh_map.items():
        if k in text:
            return v
    return None
