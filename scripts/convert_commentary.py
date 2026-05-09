# -*- coding: utf-8 -*-
"""将 ESPN commentary JSON 转换为本项目 commentary_sample.json 格式。

输入格式示例：
[
  {"event_id": 1, "time": "13'", "type": "goal", "text": "Goal! ..."}
]

输出字段：event_id, time, type, team, player, text，并为换人事件额外保留
player_in 和 player_out，便于证据链问答系统使用。
"""
import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw_commentary.json")
OUT_PATH = Path("data/commentary_sample.json")

TEAM_MAP = {
    "FC Bayern München": "拜仁慕尼黑",
    "SC Freiburg": "弗赖堡",
}


def normalize_time(t: str) -> str:
    t = str(t).strip()
    if t == "-" or not t:
        return ""
    m = re.match(r"^(\d+)'(?:\+(\d+)')?$", t)
    if m:
        minute = int(m.group(1)) + int(m.group(2) or 0)
        return f"{minute:02d}:00"
    return t


def normalize_type(event_type: str, text: str) -> str:
    if text.startswith("Attempt saved"):
        return "shot_saved"
    if text.startswith("Attempt blocked"):
        return "shot_blocked"
    if text.startswith("Attempt missed"):
        return "shot_missed"
    if text.startswith("Hand ball"):
        return "hand_ball"
    if "hits the left post" in text or "hits the right post" in text:
        return "shot_post"
    return event_type


def extract_team(text: str) -> str:
    for eng, zh in TEAM_MAP.items():
        if f"({eng})" in text or text.startswith(f"Corner, {eng}") or text.startswith(f"Substitution, {eng}"):
            return zh
    return ""


def extract_player(text: str, event_type: str):
    if event_type == "substitution":
        m = re.search(r"Substitution, .*?\.\s*([^\.]+?)\s+replaces\s+([^\.]+?)(?:\s+because of an injury)?\.", text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    patterns = [
        r"Goal! .*?\.\s*([^\(\.]+?)\s*\(",
        r"Attempt (?:missed|blocked|saved)\.\s*([^\(\.]+?)\s*\(",
        r"Foul by\s+([^\(\.]+?)\s*\(",
        r"([^\(\.]+?)\s*\([^\)]*\) wins a free kick",
        r"([^\(\.]+?)\s*\([^\)]*\) is shown",
        r"Hand ball by\s+([^\(\.]+?)\s*\(",
        r"([^\(\.]+?)\s*\([^\)]*\) hits the",
        r"Offside, .*?\.\s*([^\.]+?) tries a through ball",
        r"Corner, .*?\. Conceded by\s+([^\.]+?)\.",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip(), ""
    return "", ""


def translate_text(text: str) -> str:
    out = text
    for eng, zh in TEAM_MAP.items():
        out = out.replace(eng, zh)
    replacements = {
        "Goal!": "进球！",
        "First Half begins.": "上半场开始。",
        "Second Half begins": "下半场开始，",
        "First Half ends": "上半场结束，",
        "Second Half ends": "下半场结束，",
        "Match ends": "全场比赛结束，",
        "Substitution": "换人",
        "Corner": "角球",
        "Offside": "越位",
        "Foul by": "犯规球员",
        "wins a free kick": "赢得任意球",
        "is shown the yellow card for a bad foul": "因严重犯规被出示黄牌",
        "Attempt saved.": "射门被扑出。",
        "Attempt blocked.": "射门被封堵。",
        "Attempt missed.": "射门偏出。",
        "Hand ball by": "手球球员",
        "Conceded by": "由其造成：",
        "Assisted by": "助攻：",
        "because of an injury": "原因：受伤",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.replace("，,", "，").replace("，  ", "， ")


def convert(raw_events):
    converted = []
    for event in raw_events:
        original_type = event.get("type", "")
        text = event.get("text", "")
        event_type = normalize_type(original_type, text)
        player, player_out = extract_player(text, event_type)
        time = normalize_time(event.get("time", ""))
        if not time:
            if event_type == "kickoff":
                time = "00:00"
            elif event_type == "full_time":
                time = "90:00"
        item = {
            "event_id": event.get("event_id"),
            "time": time,
            "type": event_type,
            "team": extract_team(text),
            "player": player,
            "text": translate_text(text),
        }
        if event_type == "substitution":
            item["player_in"] = player
            item["player_out"] = player_out
        if original_type != event_type:
            item["original_type"] = original_type
        converted.append(item)
    return converted


def main():
    raw_events = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    converted = convert(raw_events)
    OUT_PATH.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已转换 {len(converted)} 条事件 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
