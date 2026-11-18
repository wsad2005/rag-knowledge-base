"""生成演示用样例文档 → ``samples/``。

四份文件覆盖四种解析路径，内容互相独立，便于演示“问到哪篇就引用哪篇”。

用法（在项目根目录执行，需要先建好虚拟环境并安装依赖）::

    py -3.14 -m scripts.make_samples
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from docx.shared import Pt

from scripts.pdf_factory import build_pdf

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"

REIMBURSEMENT = """员工报销管理制度

第一章 总则

第一条 为规范公司费用报销流程，明确审批责任，特制定本制度。本制度适用于公司全体正式员工、实习生与外聘顾问。

第二条 报销遵循真实、合规、及时三项原则。所有报销必须有真实业务背景，票据必须合法有效，不得拆分发票规避审批。

第二章 报销流程

第三条 报销时限。员工应在费用发生之日起 15 个工作日内提交报销申请，逾期需由部门负责人出具书面说明。
发票抬头必须与公司主体名称完全一致，否则财务将直接退回申请。

第四条 审批链路。单笔金额 5000 元以下的报销由直属主管审批；5000 元及以上需财务总监复核；
20000 元以上还需总经理审批。审批在系统中自动流转，任一环节驳回都会通过邮件通知申请人，申请人修改后可重新提交。

第五条 付款周期。财务每周三、周五各安排一次付款，审批通过且票据核验无误的报销单，一般在 3 个工作日内到账。

第三章 差旅标准

第六条 住宿标准。一线城市每晚上限 600 元，二线城市每晚上限 400 元，其他城市每晚上限 300 元。
住宿费按实际发生金额报销，超标部分需提前提交特批申请，未经批准的超标费用不予报销。

第七条 交通标准。市内交通优先选择公共交通，实报实销；因公打车需在报销单中注明起止地点与事由。
高铁二等座、经济舱为默认标准，商务座与公务舱需提前审批。

第八条 餐补标准。出差期间每人每天餐补 100 元，按实际出差天数计算，无需提供餐饮发票。

第四章 特殊情形

第九条 因公受伤或突发疾病产生的医疗费用，凭二级以上医院票据全额报销，不受上述标准限制。

第十条 本制度自发布之日起施行，由财务部负责解释。与原制度不一致的，以本制度为准。
"""

RAG_NOTES = """# RAG 技术说明

## 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）把「检索」和「生成」两步组合起来：
先用向量检索从私有文档中找到与问题相关的片段，再把这些片段作为上下文交给大模型生成答案。

相比直接微调模型，RAG 的优势是知识可以随时更新、答案可以给出引用来源、成本更低。

## 完整流程

一次问答包含六个步骤：

1. **文档解析**：把 PDF、DOCX、TXT、Markdown 转成纯文本，并保留页码、章节等来源信息。
2. **文本切分**：按段落和句子边界切成语义完整的块（chunk），块之间保留一定重叠。
3. **向量化**：用 Embedding 模型把每个块编码成固定维度的向量。
4. **向量存储**：把向量与元数据写入向量数据库，建立可持久化的索引。
5. **检索**：把用户问题编码成向量，按余弦相似度召回 Top-K 个最相关的块。
6. **生成**：把问题和召回内容按固定模板拼成 Prompt，交给大模型生成带引用的答案。

## 关键参数

| 参数 | 作用 | 常见取值 |
| --- | --- | --- |
| chunk_size | 单个文本块的长度 | 300 ~ 800 字 |
| chunk_overlap | 相邻块的重叠长度 | chunk_size 的 10% ~ 20% |
| top_k | 召回条数 | 3 ~ 8 |
| score_threshold | 相似度下限 | 0.2 ~ 0.5 |

## 常见问题

### 召回不准

优先检查切分是否破坏了语义，其次检查 Embedding 模型是否适合中文。
纯字面匹配的哈希向量无法处理同义词，此时应换用语义模型。

### 答案编造

在系统提示中明确要求「只依据资料回答」，并要求标注引用编号；
同时把相似度过低的结果直接过滤掉，避免把无关内容塞进上下文。
"""

PRODUCT_DOC_TITLE = "智能知识库产品需求说明"
PRODUCT_PARAGRAPHS = [
    ("背景与目标", "企业内部的制度、手册、项目文档分散在网盘与聊天记录中，检索效率低。"
     "本产品希望把散落的文档集中到一个可问答的知识库中，让员工用自然语言就能找到答案。"),
    ("目标用户", "主要用户是人事、财务、法务等需要频繁查询制度的职能人员，以及需要快速上手新项目的新员工。"),
    ("核心功能", "文档上传与解析、自动切分与向量化、基于语义的检索、带引用来源的问答、检索结果调试面板。"),
    ("非功能需求", "单个文件上传不超过 20MB；问答首字返回时间不超过 5 秒；未配置大模型时仍可完成文档处理与检索演示。"),
]
PRODUCT_TABLE_HEADER = ("模块", "优先级", "验收标准")
PRODUCT_TABLE_ROWS = [
    ("文档管理", "P0", "支持四种格式，重复文件不重复入库"),
    ("检索", "P0", "Top-K 可调，结果携带文档名与页码"),
    ("问答", "P0", "答案必须标注引用，检索不到时明确说明"),
]

PDF_PAGES = [
    [
        "Retrieval-Augmented Generation: Reading Notes",
        "RAG couples a parametric language model with a non-parametric retriever.",
        "The retriever returns passages; the generator conditions on them.",
    ],
    [
        "Evaluation",
        "Retrieval quality is measured by recall at k and mean reciprocal rank.",
        "Generation quality is measured by answer faithfulness and citation accuracy.",
        "Ablations show that chunk overlap mainly helps multi-hop questions.",
    ],
]


def write_text() -> Path:
    path = SAMPLES_DIR / "员工报销制度.txt"
    path.write_text(REIMBURSEMENT, encoding="utf-8")
    return path


def write_markdown() -> Path:
    path = SAMPLES_DIR / "RAG技术说明.md"
    path.write_text(RAG_NOTES, encoding="utf-8")
    return path


def write_docx() -> Path:
    document = DocxDocument()
    document.add_heading(PRODUCT_DOC_TITLE, level=1)
    for heading, body in PRODUCT_PARAGRAPHS:
        document.add_heading(heading, level=2)
        paragraph = document.add_paragraph(body)
        paragraph.paragraph_format.space_after = Pt(6)

    document.add_heading("功能清单", level=2)
    table = document.add_table(rows=1 + len(PRODUCT_TABLE_ROWS), cols=3)
    table.style = "Table Grid"
    for index, value in enumerate(PRODUCT_TABLE_HEADER):
        table.cell(0, index).text = value
    for row_index, row in enumerate(PRODUCT_TABLE_ROWS, start=1):
        for column_index, value in enumerate(row):
            table.cell(row_index, column_index).text = value

    path = SAMPLES_DIR / "产品需求说明.docx"
    document.save(path)
    return path


def write_pdf() -> Path:
    path = SAMPLES_DIR / "rag-paper-notes.pdf"
    path.write_bytes(build_pdf(PDF_PAGES))
    return path


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for path in (write_text(), write_markdown(), write_docx(), write_pdf()):
        print(f"已生成 {path.relative_to(ROOT)}（{path.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
