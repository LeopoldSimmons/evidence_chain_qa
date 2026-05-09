# -*- coding: utf-8 -*-
"""可选网页界面。运行：streamlit run streamlit_app.py"""
from pathlib import Path

import streamlit as st

from src.qa_system import EvidenceChainQA

BASE_DIR = Path(__file__).resolve().parent
COMMENTARY_PATH = BASE_DIR / "data" / "commentary_sample.json"

st.set_page_config(page_title="证据链问答系统", layout="wide")
st.title("基于证据链检索增强的足球赛事长文本问答系统")
st.caption("当前样例：德甲第10轮 拜仁慕尼黑 5-0 弗赖堡。输入复杂问题，系统返回答案、意图类别与可解释证据链。")

@st.cache_resource
def load_system():
    return EvidenceChainQA(str(COMMENTARY_PATH))

system = load_system()
examples = [
    "拜仁第三个进球前，同一分钟出现了哪次更早的关键机会？这两条证据连起来说明了什么？",
    "Michael Gregoritsch这名球员的事件链有什么特点？请结合他早期参与进攻、犯规、吃牌和被换下的证据回答。",
    "弗赖堡在下半场刚开始是否曾尝试反扑？请找出46到47分钟之间能支持这一判断的连续证据。",
    "弗赖堡为什么可以被认为在50到51分钟出现了短时间纪律崩盘？",
    "拜仁的第五个进球和76分钟的换人调整之间有什么直接关系？",
]
question = st.selectbox("示例问题", examples)
custom_question = st.text_input("也可以输入自定义问题", value=question)

if st.button("开始问答"):
    result = system.answer(custom_question)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("系统输出")
        st.write("**意图类别：**", result["intent"])
        st.write("**答案：**", result["answer"])
        st.write("**证据编号：**", result["evidence_ids"])
    with right:
        st.subheader("证据链")
        for e in result["evidence"]:
            st.info(f"[{e['event_id']}] {e['time']} | {e['type']} | {e.get('team','')} | {e.get('player','')} | {e['text']}")
