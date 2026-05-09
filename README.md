# 基于证据链检索增强的足球赛事长文本问答系统

本项目是《自然语言处理》课程报告的配套代码。系统以足球比赛 commentary 为输入，实现事件检索、证据链抽取和解释式问答。当前样例数据已替换为：**德甲第10轮 拜仁慕尼黑 5-0 弗赖堡**。

## 1. 项目结构

```text
evidence_chain_qa_project/
├── data/
│   ├── raw_commentary.json        # 原始 ESPN commentary 数据
│   ├── commentary_sample.json     # 转换后的课程作业样例数据
│   └── qa_gold.json               # 复杂证据链问答标注集
├── scripts/
│   └── convert_commentary.py      # 原始 commentary 转换脚本
├── src/
│   ├── bm25.py                    # 无第三方依赖 BM25 检索
│   ├── evaluator.py               # 证据链 F1 和答案匹配评估
│   ├── qa_system.py               # 问答系统主逻辑
│   └── text_utils.py              # 文本标准化与查询扩展
├── run_demo.py                    # 命令行演示入口
├── streamlit_app.py               # 可选网页演示入口
└── requirements.txt
```

## 2. 数据格式

`commentary_sample.json` 中每条事件包含：

```json
{
  "event_id": 47,
  "time": "52:00",
  "type": "goal",
  "team": "拜仁慕尼黑",
  "player": "Leroy Sané",
  "text": "进球！ 拜仁慕尼黑 3, 弗赖堡 0. Leroy Sané ... 助攻： Eric Choupo-Moting."
}
```

换人事件额外保留：

```json
{
  "player_in": "Marcel Sabitzer",
  "player_out": "Leon Goretzka"
}
```

## 3. 运行环境

推荐 Python 3.9 及以上。命令行版本不需要额外依赖；网页版本需要安装 `streamlit`。

```bash
pip install -r requirements.txt
```

## 4. 运行命令行 Demo

```bash
python run_demo.py
```

运行后会输出：

- 问题
- 意图类型
- 系统答案
- 证据编号
- 证据链文本
- 评估结果

输出文件保存在 `outputs/`：

```text
outputs/demo_predictions.json
outputs/evaluation.json
```

## 5. 重新转换原始数据

如需从原始 ESPN commentary 文件重新生成 `commentary_sample.json`：

```bash
python scripts/convert_commentary.py
```

## 6. 运行网页界面

```bash
streamlit run streamlit_app.py
```

网页打开后，可以输入复杂问题，例如：

```text
拜仁第三个进球前，同一分钟出现了哪次更早的关键机会？这两条证据连起来说明了什么？
```

## 7. 当前 qa_gold.json 的设计原则

`qa_gold.json` 中的问题尽量避免简单比分事实，主要覆盖：

1. 时间窗口推理，例如 46 到 47 分钟的连续反扑；
2. 球员事件链，例如 Michael Gregoritsch 的助攻、犯规、黄牌和被换下；
3. 换人到进球的因果链，例如 Marcel Sabitzer 76 分钟登场、80 分钟进球；
4. 多事件集合推理，例如 5 个进球的射门方式和助攻者。

这类问题要求系统从多条 commentary 事件中组合证据，因此更适合体现“证据发现能力”。

## 补充统计图表

报告中新增的事件类型分布、问题类型分布、证据链长度分布和消融实验图表由以下脚本生成：

```bash
python scripts/extra_analysis.py
```

生成结果位于 `outputs/` 目录，包括：

- `event_type_distribution.png`
- `qa_type_distribution.png`
- `evidence_length_distribution.png`
- `ablation_comparison.png`
- `extra_stats.json`
- `ablation_results.json`


图表配色采用课程报告指定配色：#2B307A, #77C2F3, #F7EEF6, #D8A0C7, #A96FB0。
