"""One-time, fail-closed import of legacy JSON persistence into SQLite."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.storage.database import utc_now
from backend.storage.repository import StorageRepository


LEGACY_MIGRATION_KEY = "legacy_json_v1"


class LegacyMigrationError(RuntimeError):
    pass


class LegacyMigrator:
    def __init__(self, repository: StorageRepository, base_dir: Path | str, chroma_dir: Path | str) -> None:
        self.repository = repository
        self.base_dir = Path(base_dir).expanduser().resolve()
        raw_chroma = Path(chroma_dir).expanduser()
        self.chroma_dir = raw_chroma.resolve() if raw_chroma.is_absolute() else (self.base_dir / raw_chroma).resolve()
        self.data_dir = self.repository.database.path.parent
        self.report_dir = self.data_dir / "migration_reports"
        self.backup_dir = self.data_dir / "legacy_backups"

    def migrate_once(self) -> Dict[str, Any]:
        previous = self.repository.migration_run(LEGACY_MIGRATION_KEY)
        files = self._legacy_files()
        if previous and previous.get("status") == "success":
            # A fresh install may start before legacy files are copied in. Only that
            # empty-source case is reopened; migrated non-empty sources never auto-import again.
            if previous.get("source_hash") or not files:
                return {**previous, "status": "already_migrated"}
        if not files:
            now = utc_now()
            self.repository.record_migration_run(
                LEGACY_MIGRATION_KEY, "", "success", detail="未发现旧版 JSON 数据", started_at=now, completed_at=now,
            )
            return {"status": "success", "message": "未发现旧版 JSON 数据", "counts": {}}

        started_at = utc_now()
        source_manifest = self._manifest(files)
        source_hash = self._manifest_hash(source_manifest)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_root = self.backup_dir / stamp
        report_path = self.report_dir / f"legacy-{stamp}.json"
        report = {
            "migration_key": LEGACY_MIGRATION_KEY,
            "status": "started",
            "started_at": started_at,
            "source_hash": source_hash,
            "sources": source_manifest,
            "backup_dir": str(backup_root),
            "report_path": str(report_path),
            "warnings": [],
            "counts": {},
        }
        self.report_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._backup_files(files, backup_root)
            snapshot = self._build_snapshot(report)
            expected_counts = self._expected_counts(snapshot)
            snapshot["_expected_counts"] = expected_counts
            counts = self.repository.import_legacy_snapshot(snapshot)
            reconciliation = {
                key: {
                    "expected": expected_counts[key],
                    "imported": counts.get(key, 0),
                    "matched": expected_counts[key] == counts.get(key, 0),
                }
                for key in expected_counts
            }
            if not all(item["matched"] for item in reconciliation.values()):
                raise ValueError("迁移数量对账失败")
            report.update({
                "status": "success",
                "completed_at": utc_now(),
                "counts": counts,
                "reconciled": True,
                "reconciliation": reconciliation,
            })
            self._write_report(report_path, report)
            self.repository.record_migration_run(
                LEGACY_MIGRATION_KEY, source_hash, "success", str(report_path),
                detail="旧版 JSON 已导入 SQLite；旧文件仅保留为迁移备份，不再双写",
                started_at=started_at, completed_at=report["completed_at"],
            )
            return report
        except Exception as exc:
            report.update({"status": "failed", "completed_at": utc_now(), "error": str(exc)[:1000]})
            self._write_report(report_path, report)
            self.repository.record_migration_run(
                LEGACY_MIGRATION_KEY, source_hash, "failed", str(report_path), str(exc),
                started_at=started_at, completed_at=report["completed_at"],
            )
            raise LegacyMigrationError(f"旧数据迁移失败，已保留备份和报告: {report_path}: {exc}") from exc

    def _legacy_files(self) -> List[Tuple[str, Path]]:
        candidates = (
            ("categories", self.base_dir / "categories.json"),
            ("vector_store", self.chroma_dir / "vector_store.json"),
            ("project_context", self.chroma_dir / "project_context.json"),
            ("feishu_workspace", self.chroma_dir / "feishu_workspace.json"),
            ("handovers", self.chroma_dir / "resignations.json"),
            ("graph_state", self.base_dir / "knowledge_graph_state.json"),
            ("graph_import", self.base_dir / "knowledge_graph_import.json"),
        )
        # config.json intentionally stays outside migration and backup because it contains secrets.
        return [(name, path) for name, path in candidates if path.is_file()]

    def _manifest(self, files: Iterable[Tuple[str, Path]]) -> List[Dict[str, Any]]:
        result = []
        for name, path in files:
            result.append({
                "name": name,
                "path": str(path),
                "relative_path": self._relative_name(path),
                "size": path.stat().st_size,
                "sha256": self._file_hash(path),
            })
        uploads = self.base_dir / "uploads"
        if uploads.is_dir():
            upload_files = sorted(path for path in uploads.rglob("*") if path.is_file())
            result.append({
                "name": "uploads_inventory",
                "path": str(uploads),
                "relative_path": "uploads/",
                "file_count": len(upload_files),
                "size": sum(path.stat().st_size for path in upload_files),
                "sha256": self._directory_hash(upload_files),
                "copied": False,
            })
        return result

    @staticmethod
    def _manifest_hash(manifest: List[Dict[str, Any]]) -> str:
        normalized = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _backup_files(self, files: Iterable[Tuple[str, Path]], backup_root: Path) -> None:
        backup_root.mkdir(parents=True, exist_ok=False)
        try:
            backup_root.chmod(0o700)
        except OSError:
            pass
        for _, path in files:
            target = backup_root / self._relative_name(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            try:
                target.chmod(0o600)
            except OSError:
                pass
        manifest = self._manifest(files)
        manifest_path = backup_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            manifest_path.chmod(0o600)
        except OSError:
            pass

    def _relative_name(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return path.name

    def _build_snapshot(self, report: Dict[str, Any]) -> Dict[str, Any]:
        categories = self._read_optional("categories", [], allow_legacy_encoding=True)
        if not isinstance(categories, list):
            raise ValueError("categories.json 必须是数组")
        vector = self._read_optional("vector_store", {})
        context = self._read_optional("project_context", {})
        feishu = self._read_optional("feishu_workspace", {})
        handovers = self._read_optional("handovers", [])
        graph_state = self._read_optional("graph_state", {})
        graph_import = self._read_optional("graph_import", {})

        documents, document_categories = self._normalize_documents(vector, context)
        merged_categories = list(dict.fromkeys(
            [str(item).strip() for item in categories if str(item).strip()] + document_categories
        ))
        graph = self._normalize_graph(graph_state, graph_import)
        if not isinstance(feishu, dict):
            raise ValueError("feishu_workspace.json 必须是对象")
        if not isinstance(handovers, list):
            raise ValueError("resignations.json 必须是数组")
        report["warnings"].append("config.json 含密钥，按安全规则未复制、未导入数据库")
        uploads = self.base_dir / "uploads"
        return {
            "project_name": str(context.get("project_name") or "默认项目") if isinstance(context, dict) else "默认项目",
            "categories": merged_categories,
            "documents": documents,
            "handovers": handovers,
            "feishu": {
                "version": int(feishu.get("version") or 4),
                "groups": self._dict_list(feishu.get("groups")),
                "messages": self._dict_list(feishu.get("messages")),
                "candidates": self._dict_list(feishu.get("candidates")),
                "assets": self._dict_list(feishu.get("assets")),
                "audit": self._dict_list(feishu.get("audit")),
                "diagnostics": feishu.get("diagnostics") if isinstance(feishu.get("diagnostics"), dict) else {},
                "updated_at": str(feishu.get("updated_at") or ""),
            },
            "graph": graph,
            "upload_files": len([path for path in uploads.rglob("*") if path.is_file()]) if uploads.is_dir() else 0,
        }

    @staticmethod
    def _expected_counts(snapshot: Dict[str, Any]) -> Dict[str, int]:
        feishu = snapshot.get("feishu") or {}
        graph = snapshot.get("graph") or {}
        node_ids = {
            str(item.get("id")) for item in graph.get("nodes", [])
            if isinstance(item, dict) and item.get("id")
        }
        valid_edges = [
            item for item in graph.get("edges", [])
            if isinstance(item, dict)
            and str(item.get("source") or item.get("from") or "") in node_ids
            and str(item.get("target") or item.get("to") or "") in node_ids
        ]
        return {
            "categories": len(snapshot.get("categories", [])),
            "documents": len(snapshot.get("documents", [])),
            "chunks": sum(len(item.get("chunks", [])) for item in snapshot.get("documents", [])),
            "handovers": len(snapshot.get("handovers", [])),
            "feishu_groups": len([item for item in feishu.get("groups", []) if item.get("chat_id")]),
            "feishu_messages": len(feishu.get("messages", [])),
            "feishu_candidates": len([
                item for item in feishu.get("candidates", []) if item.get("candidate_id")
            ]),
            "feishu_assets": len([item for item in feishu.get("assets", []) if item.get("asset_id")]),
            "audit_events": len(feishu.get("audit", [])[-2000:]),
            "graph_nodes": len(node_ids),
            "graph_edges": len(valid_edges),
            "upload_files": int(snapshot.get("upload_files") or 0),
        }

    def _read_optional(self, name: str, default: Any, allow_legacy_encoding: bool = False) -> Any:
        path = dict(self._legacy_files()).get(name)
        if not path:
            return default
        raw = path.read_bytes()
        encodings = ("utf-8-sig", "gb18030", "gbk") if allow_legacy_encoding else ("utf-8-sig",)
        last_error: Optional[Exception] = None
        for encoding in encodings:
            try:
                return json.loads(raw.decode(encoding))
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} JSON 损坏: {exc}") from exc
        raise ValueError(f"{path.name} 编码无法识别: {last_error}")

    def _normalize_documents(self, vector: Any, context: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        if vector and not isinstance(vector, dict):
            raise ValueError("vector_store.json 必须是对象")
        vector = vector if isinstance(vector, dict) else {}
        texts = vector.get("texts", []) or []
        metadatas = vector.get("metadatas", []) or []
        vector_ids = vector.get("ids", []) or []
        vectors = vector.get("vectors", []) or []
        lengths = {len(texts), len(metadatas), len(vector_ids), len(vectors)}
        if vector and len(lengths) != 1:
            raise ValueError(
                "vector_store.json 数组长度不一致: "
                f"texts={len(texts)}, metadatas={len(metadatas)}, ids={len(vector_ids)}, vectors={len(vectors)}"
            )
        context_metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
        if context_metadata and not isinstance(context_metadata, dict):
            raise ValueError("project_context.json metadata 必须是对象")
        grouped: Dict[str, Dict[str, Any]] = {}
        categories: List[str] = []
        for index, text in enumerate(texts):
            metadata = dict(metadatas[index]) if isinstance(metadatas[index], dict) else {}
            stored_file = str(metadata.get("stored_file") or metadata.get("source_file") or f"legacy-{index}")
            item = grouped.setdefault(stored_file, {"document": dict(metadata), "chunks": []})
            item["document"].update({
                "stored_file": stored_file,
                "source_file": str(metadata.get("source_file") or stored_file),
            })
            item["chunks"].append({
                "chunk_index": len(item["chunks"]),
                "vector_id": str(vector_ids[index]),
                "text": str(text or ""),
                "metadata": metadata,
            })
        for stored_file, item in grouped.items():
            context_item = context_metadata.get(stored_file, {}) if isinstance(context_metadata, dict) else {}
            if not context_item:
                context_item = next((
                    value for value in context_metadata.values()
                    if isinstance(value, dict) and (
                        value.get("stored_file") == stored_file
                        or value.get("source_file") == item["document"].get("source_file")
                    )
                ), {})
            item["document"] = {**item["document"], **(context_item if isinstance(context_item, dict) else {})}
            item_categories = item["document"].get("categories") or (
                [item["document"].get("category")] if item["document"].get("category") else []
            )
            item["document"]["categories"] = item_categories or ["未分类"]
            item["document"]["category"] = item["document"].get("category") or item["document"]["categories"][0]
            categories.extend(str(value) for value in item["document"]["categories"] if value)
        for context_key, context_item in context_metadata.items():
            if not isinstance(context_item, dict):
                continue
            stored_file = str(context_item.get("stored_file") or context_key or context_item.get("source_file") or "").strip()
            if not stored_file or stored_file in grouped:
                continue
            document = {
                **context_item,
                "stored_file": stored_file,
                "source_file": str(context_item.get("source_file") or stored_file),
            }
            item_categories = document.get("categories") or (
                [document.get("category")] if document.get("category") else []
            )
            document["categories"] = item_categories or ["未分类"]
            document["category"] = document.get("category") or document["categories"][0]
            grouped[stored_file] = {"document": document, "chunks": []}
            categories.extend(str(value) for value in document["categories"] if value)
        return list(grouped.values()), list(dict.fromkeys(categories))

    @staticmethod
    def _normalize_graph(state: Any, imported: Any) -> Dict[str, Any]:
        if state and not isinstance(state, dict):
            raise ValueError("knowledge_graph_state.json 必须是对象")
        if imported and not isinstance(imported, dict):
            raise ValueError("knowledge_graph_import.json 必须是对象")
        state = state if isinstance(state, dict) else {}
        imported = imported if isinstance(imported, dict) else {}
        imported_node_ids = {str(item.get("id")) for item in imported.get("nodes", []) if isinstance(item, dict) and item.get("id")}
        imported_edge_ids = {str(item.get("id")) for item in imported.get("edges", []) if isinstance(item, dict) and item.get("id")}
        nodes: Dict[str, Dict[str, Any]] = {}
        for raw in [*state.get("nodes", []), *imported.get("nodes", [])]:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            node = dict(raw)
            node_id = str(node["id"])
            if node_id in imported_node_ids:
                node["origin"] = "imported"
            nodes[node_id] = {**nodes.get(node_id, {}), **node}
        edges: Dict[str, Dict[str, Any]] = {}
        for raw in [*state.get("edges", []), *imported.get("edges", [])]:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source") or raw.get("from") or "")
            target = str(raw.get("target") or raw.get("to") or "")
            if not source or not target:
                continue
            edge_id = str(raw.get("id") or hashlib.sha256(f"{source}:{target}:{raw.get('label')}".encode()).hexdigest()[:24])
            edge = {**raw, "id": edge_id, "source": source, "target": target}
            if edge_id in imported_edge_ids or source in imported_node_ids or target in imported_node_ids:
                edge.update({"method": "imported", "auto_generated": False, "protected": True})
            edges[edge_id] = {**edges.get(edge_id, {}), **edge}
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    @staticmethod
    def _dict_list(value: Any) -> List[Dict[str, Any]]:
        return [dict(item) for item in value] if isinstance(value, list) else []

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _directory_hash(cls, files: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.name).encode("utf-8"))
            digest.update(cls._file_hash(path).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _write_report(path: Path, report: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
