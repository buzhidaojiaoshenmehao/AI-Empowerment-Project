"""文档加载器 —— 支持 PDF、TXT、MD、DOCX、CSV、JSON（纯 Python，无 heavy 依赖）"""
import csv
import json
import os
from typing import List

from langchain_core.documents import Document


def _load_text_file(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """通用文本文件加载（TXT、MD）"""
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        text = f.read()
    file_name = os.path.basename(file_path)
    return [Document(page_content=text, metadata={"source_file": file_name, "source": file_name})]


def _load_json_file(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """读取 JSON 并转为稳定、可检索的文本。"""
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        raw = f.read()
    file_name = os.path.basename(file_path)
    try:
        parsed = json.loads(raw)
        text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        text = raw
    return [Document(page_content=text, metadata={"source_file": file_name, "source": file_name})]


def _load_pdf_file(file_path: str) -> List[Document]:
    """按页读取 PDF，保留旧加载器使用的零起始页码元数据。"""
    from pypdf import PdfReader

    file_name = os.path.basename(file_path)
    reader = PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source_file": file_name, "source": file_name, "page": page_index},
        )
        for page_index, page in enumerate(reader.pages)
    ]


def _load_docx_file(file_path: str) -> List[Document]:
    """读取 DOCX 文本，不依赖停止维护的 langchain-community。"""
    import docx2txt

    file_name = os.path.basename(file_path)
    text = docx2txt.process(file_path) or ""
    return [Document(page_content=text, metadata={"source_file": file_name, "source": file_name})]


def _load_csv_file(file_path: str) -> List[Document]:
    """按行读取 CSV，并保持每行一个可检索 Document。"""
    file_name = os.path.basename(file_path)
    documents: List[Document] = []
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            content = "\n".join(f"{key}: {value or ''}" for key, value in row.items() if key is not None)
            documents.append(Document(
                page_content=content,
                metadata={"source_file": file_name, "source": file_name, "row": row_index},
            ))
    return documents


def load_document(file_path: str, filename_hint: str = "") -> List[Document]:
    """根据文件扩展名选择加载器，返回 Document 列表"""
    ext = os.path.splitext(filename_hint or file_path)[1].lower()

    if ext == ".pdf":
        docs = _load_pdf_file(file_path)
    elif ext in (".txt", ".md"):
        docs = _load_text_file(file_path)
    elif ext == ".json":
        docs = _load_json_file(file_path)
    elif ext == ".docx":
        docs = _load_docx_file(file_path)
    elif ext == ".csv":
        docs = _load_csv_file(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    # 为每个文档块附加元数据（来源文件名）
    file_name = os.path.basename(filename_hint or file_path)
    for doc in docs:
        doc.metadata["source_file"] = file_name
        doc.metadata["source"] = file_name

    return docs
