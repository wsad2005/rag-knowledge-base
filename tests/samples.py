"""四种格式的测试样本（全部在内存中构造，不依赖仓库里的二进制文件）。"""

from __future__ import annotations

from io import BytesIO

from docx import Document as DocxDocument

from scripts.pdf_factory import build_pdf

TXT_CONTENT = """员工报销管理制度

第一条 报销时限。员工应在费用发生之日起 15 个工作日内提交报销申请，逾期需要部门负责人书面说明。
发票抬头必须与公司主体一致，否则财务将退回申请。

第二条 审批流程。单笔金额 5000 元以下由直属主管审批，5000 元及以上需财务总监复核。
审批链路在系统中自动流转，任何环节驳回都会通知申请人。

第三条 差旅标准。一线城市住宿标准为每晚 600 元，二线城市为每晚 400 元，其他城市为 300 元。
超标部分需要提前提交特批申请，未经批准的超标费用不予报销。

第四条 特殊情况。因公受伤或突发疾病产生的医疗费用，凭医院票据全额报销，不受上述标准限制。
"""

MD_CONTENT = """# 项目部署手册

## 环境准备

需要 Python 3.11 及以上版本，并准备一个可写的 `data` 目录。

- 安装依赖：`pip install -r requirements.txt`
- 配置环境变量：复制 `.env.example` 为 `.env`

## 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000` 即可看到页面。

## 常见问题

### 端口被占用

使用 `--port 8001` 换一个端口，或用 `netstat -ano` 找到占用进程。

### 向量库初始化失败

确认 `data/chroma` 目录有写权限，并且没有其他进程正在使用该目录。
"""

DOCX_HEADING = "产品需求说明"
DOCX_PARAGRAPH = "本系统需要支持文档上传、解析、切分与向量检索，并提供基于大模型的问答能力。"
DOCX_TABLE_ROW = ("模块", "优先级", "说明")
DOCX_TABLE_CELLS = ("检索", "高", "支持 Top-K 与相似度阈值")


def docx_bytes() -> bytes:
    """构造一个包含标题、正文、表格的 DOCX。"""
    document = DocxDocument()
    document.add_heading(DOCX_HEADING, level=1)
    document.add_paragraph(DOCX_PARAGRAPH)
    document.add_heading("功能清单", level=2)
    table = document.add_table(rows=2, cols=3)
    for index, value in enumerate(DOCX_TABLE_ROW):
        table.cell(0, index).text = value
    for index, value in enumerate(DOCX_TABLE_CELLS):
        table.cell(1, index).text = value

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def pdf_bytes() -> bytes:
    """两页英文 PDF：用于验证页码保留。"""
    return build_pdf(
        [
            [
                "Retrieval Augmented Generation Overview",
                "RAG combines a retriever with a language model.",
                "The retriever selects passages from a vector store.",
            ],
            [
                "Evaluation Notes",
                "Recall at k and answer faithfulness are the key metrics.",
            ],
        ]
    )


def scanned_pdf_bytes() -> bytes:
    """无文本层的 PDF（模拟扫描件）。"""
    return build_pdf([[]])


def gbk_txt_bytes() -> bytes:
    """GBK 编码的中文文本，用于验证编码回退。"""
    return "这是一份用 GBK 编码保存的中文说明文档。\n\n第二段内容用于验证解码是否正确。".encode(
        "gb18030"
    )
