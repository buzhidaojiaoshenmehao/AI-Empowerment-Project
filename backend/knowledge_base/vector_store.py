# -*- coding: utf-8 -*-
"""向量存储"""
import os,re,json,hashlib,logging,unicodedata
from functools import wraps
from threading import RLock
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import settings

logger=logging.getLogger(__name__)
TEXT_EN=re.compile(r"[a-zA-Z]+")
TEXT_CN=re.compile(r"[\u4e00-\u9fff]+")
TEXT_NUM=re.compile(r"\d+(?:\.\d+)?")


def _normalize_text(text):
    """Normalize visually equivalent PDF/OCR compatibility characters."""
    return unicodedata.normalize("NFKC", str(text or ""))


def _synchronized(method):
    @wraps(method)
    def wrapped(self,*args,**kwargs):
        with self._lock:
            return method(self,*args,**kwargs)
    return wrapped



class LocalEmbeddings(Embeddings):
    EMBEDDING_DIM = 512
    def __init__(self):pass
    def _tokenize(self,text):
        text=_normalize_text(text)
        tokens=[]
        for w in TEXT_EN.findall(text.lower()):
            if len(w)>1:tokens.append("en:"+w)
        for cb in TEXT_CN.findall(text):
            for i in range(len(cb)):
                tokens.append("c:"+cb[i])
                if i>0:tokens.append("b:"+cb[i-1:i+1])
        for n in TEXT_NUM.findall(text):tokens.append("n:"+n)
        return tokens
    def _embed(self,text):
        tokens=self._tokenize(text)
        vec=np.zeros(self.EMBEDDING_DIM,dtype=np.float32)
        for t in tokens:
            h=hashlib.md5(t.encode()).hexdigest()
            i1=int(h[:8],16)%self.EMBEDDING_DIM
            i2=int(h[8:16],16)%self.EMBEDDING_DIM
            s=1 if int(h[16:20],16)%2==0 else -1
            vec[i1]+=s;vec[i2]+=s
        nrm=np.linalg.norm(vec)
        return (vec/nrm).tolist() if nrm>0 else vec.tolist()
    def embed_documents(self,texts):return[self._embed(t) for t in texts]
    def embed_query(self,text):return self._embed(text)


def _create_embeddings():
    base_url = (settings.OPENAI_BASE_URL or "").lower()
    if settings.OPENAI_API_KEY and "openai" in base_url:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL,openai_api_key=settings.OPENAI_API_KEY,openai_api_base=settings.OPENAI_BASE_URL)
        except Exception as exc:
            logger.warning("remote embeddings unavailable, using local hybrid retrieval: %s", exc)
    return LocalEmbeddings()


class VectorStore:
    """纯 numpy 向量数据库"""
    def __init__(self):
        self._embeddings=None
        self._splitter=None
        self._vecs=[]
        self._texts=[]
        self._metas=[]
        self._ids=[]
        self._loaded=False
        self._lock=RLock()
        self._saved_index_fingerprint=""
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings=_create_embeddings()
        return self._embeddings
    @property
    def splitter(self):
        if self._splitter is None:
            self._splitter=RecursiveCharacterTextSplitter(
                chunk_size=max(100,int(settings.CHUNK_SIZE)),
                chunk_overlap=max(0,min(int(settings.CHUNK_OVERLAP),int(settings.CHUNK_SIZE)-1)),
                separators=["\n\n","\n","。","；","！","？",". ","; "," ",""],
            )
        return self._splitter

    def split_documents(self,docs):
        """Split extracted documents into stable retrieval and graph units.

        JSON graph imports remain intact because their parser requires the full
        payload. Page/row metadata is preserved on every derived chunk.
        """
        chunks=[]
        for source_index,document in enumerate(docs or []):
            metadata=dict(getattr(document,"metadata",{}) or {})
            content=str(getattr(document,"page_content","") or "")
            source=str(metadata.get("source_file") or metadata.get("source") or "").lower()
            if source.endswith(".json") or len(content)<=max(100,int(settings.CHUNK_SIZE)):
                pieces=[Document(page_content=content,metadata=metadata)]
            else:
                pieces=self.splitter.split_documents([
                    Document(page_content=content,metadata=metadata)
                ])
            for local_index,piece in enumerate(pieces):
                piece.metadata=dict(piece.metadata or {})
                piece.metadata["source_part_index"]=source_index
                piece.metadata["source_part_chunk_index"]=local_index
                chunks.append(piece)
        for chunk_index,chunk in enumerate(chunks):
            chunk.metadata["chunk_index"]=chunk_index
            chunk.metadata["chunk_count"]=len(chunks)
        return chunks
    def _path(self):
        return Path(settings.CHROMA_PERSIST_DIR)/"vector_store.json"

    def _load(self):
        if self._loaded:return
        p=self._path()
        if p.exists():
            try:
                d=json.loads(p.read_text(encoding="utf-8"))
                self._texts=d.get("texts",[])
                self._metas=d.get("metadatas",[])
                self._ids=d.get("ids",[])
                self._vecs=[np.array(v,dtype=np.float32) for v in d.get("vectors",[])]
                self._saved_index_fingerprint=str(d.get("index_fingerprint") or "")
            except Exception as e:
                logger.warning("load failed: %s",e)
        self._loaded=True
    def _save(self):
        p=self._path();p.parent.mkdir(parents=True,exist_ok=True)
        temporary=p.with_suffix(".tmp")
        with temporary.open("w",encoding="utf-8") as handle:
            self._saved_index_fingerprint=self._current_index_fingerprint()
            json.dump({
                "index_fingerprint":self._saved_index_fingerprint,
                "vectors":[v.tolist() for v in self._vecs],"texts":self._texts,
                "metadatas":self._metas,"ids":self._ids,
            },handle,ensure_ascii=False,indent=2)
            handle.flush();os.fsync(handle.fileno())
        temporary.replace(p)

    def _current_index_fingerprint(self):
        backend = "openai" if settings.OPENAI_API_KEY and "openai" in (settings.OPENAI_BASE_URL or "").lower() else "local"
        payload = {
            # Schema 3 uses the authoritative SQLite chunk_id as the index key.
            # Historical vector_id values were content-derived and could collide
            # across document versions.
            "schema":4,
            "backend":backend,
            "text_normalization":"NFKC",
            "embedding_model":settings.EMBEDDING_MODEL if backend == "openai" else settings.LOCAL_EMBEDDING_MODEL,
            "dimension":LocalEmbeddings.EMBEDDING_DIM if backend == "local" else None,
            "chunk_size":settings.CHUNK_SIZE,
            "chunk_overlap":settings.CHUNK_OVERLAP,
        }
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _authoritative_metadata(chunk: Dict[str, Any], documents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build retrieval metadata from SQLite-owned chunk and asset fields.

        Historical chunk metadata predates the unified asset model. Parent rows
        therefore override embedded metadata so a rebuilt index cannot retain a
        stale version, lifecycle, authority or ACL description.
        """
        metadata = dict(chunk.get("metadata") or {})
        parent = documents.get(str(chunk.get("document_id") or ""), {})
        authoritative = {**parent, **{
            key: value for key, value in chunk.items()
            if key not in {"metadata", "text", "text_content"}
        }}
        for key in (
            "organization_id", "project_id", "source_id", "asset_id", "version_id",
            "document_id", "chunk_id", "chunk_index", "source_type", "source_file",
            "stored_file", "status", "asset_status", "version_status",
            "is_current_version", "valid_from", "valid_until", "review_due_at",
            "authority_level", "authority_score", "sensitivity_level", "visibility",
            "access_policy_id", "acl_revision", "owner_user_id", "category",
            "categories", "applicable_roles", "topics", "page", "section",
        ):
            if key in authoritative and authoritative[key] is not None:
                metadata[key] = authoritative[key]
        metadata["chunk_id"] = str(chunk.get("chunk_id") or metadata.get("chunk_id") or "")
        return metadata

    @staticmethod
    def _document_map(repository) -> Dict[str, Dict[str, Any]]:
        """Return accessible parent metadata when the repository exposes it.

        The chunk row remains sufficient for background projection rebuilds;
        this adapter is deliberately compatible with the current and upcoming
        unified-asset repository shapes.
        """
        try:
            documents = repository.list_documents()
        except Exception:
            return {}
        return {
            str(item.get("document_id") or ""): dict(item)
            for item in documents
            if isinstance(item, dict) and item.get("document_id")
        }

    @_synchronized
    def add_documents(self,docs):
        """Persist an authoritative chunk write and its derived index atomically.

        The JSON index is written through ``_save``'s fsync + replace sequence.
        A failed derived write is never reported as a successful vector
        projection; callers can then fail the staged asset version and invoke
        their existing authoritative compensation path.
        """
        docs=self.split_documents(docs)
        self._load()
        previous_state=(
            list(self._vecs),list(self._texts),list(self._metas),list(self._ids),
            self._saved_index_fingerprint,
        )
        texts=[d.page_content for d in docs]
        metadatas=[dict(d.metadata or {}) for d in docs]
        vecs=self.embeddings.embed_documents(texts)
        provisional_ids=[]
        for index,t in enumerate(texts):
            metadata=metadatas[index] if index<len(metadatas) else {}
            identity="\n".join([
                str(metadata.get("stored_file") or metadata.get("source_file") or ""),
                str(metadata.get("page") or index),t,
            ])
            provisional_ids.append(hashlib.sha256(identity.encode()).hexdigest()[:24])
        try:
            from backend.storage import get_repository

            stored=get_repository().upsert_document_chunks(docs,provisional_ids) or {}
            chunk_ids=list(stored.get("chunk_ids") or [])
            if len(chunk_ids)!=len(docs):
                raise RuntimeError("SQLite 返回的知识块标识数量与写入数量不一致")
            document_id=str(stored.get("document_id") or "")
            if not document_id:
                raise RuntimeError("SQLite 未返回知识块所属文档标识")
            for index,metadata in enumerate(metadatas):
                metadata["document_id"]=document_id
                metadata["chunk_id"]=str(chunk_ids[index])

            # An upsert may replace chunks for the same document. Remove any
            # matching index entries before appending the new authoritative IDs.
            replaced=set(str(item) for item in chunk_ids)
            kept=[
                (vector,text,metadata,chunk_id)
                for vector,text,metadata,chunk_id in zip(self._vecs,self._texts,self._metas,self._ids)
                if str(chunk_id) not in replaced
            ]
            if kept:
                self._vecs,self._texts,self._metas,self._ids=map(list,zip(*kept))
            else:
                self._vecs=[];self._texts=[];self._metas=[];self._ids=[]
            self._vecs.extend(np.array(vector,dtype=np.float32) for vector in vecs)
            self._texts.extend(texts)
            self._metas.extend(metadatas)
            self._ids.extend(chunk_ids)
            self._save()
        except Exception:
            (
                self._vecs,self._texts,self._metas,self._ids,
                self._saved_index_fingerprint,
            )=previous_state
            raise
        return len(docs)

    @_synchronized
    def reconcile_with_storage(self):
        """Repair the derived vector file from authoritative SQLite chunks."""
        self._load()
        from backend.storage import get_repository

        repository=get_repository()
        chunks=repository.document_chunks()
        documents=self._document_map(repository)
        required={str(item.get("chunk_id")):item for item in chunks if item.get("chunk_id")}
        seen=set();kept=[];metadata_changed=False
        fingerprint_changed=self._saved_index_fingerprint!=self._current_index_fingerprint()
        index_shape_changed=not (
            len(self._vecs)==len(self._texts)==len(self._metas)==len(self._ids)
        )
        if not fingerprint_changed:
            for vector,text,metadata,vector_id in zip(self._vecs,self._texts,self._metas,self._ids):
                key=str(vector_id)
                if key in required and key not in seen:
                    item=required[key]
                    authoritative_text=str(item.get("text") or "")
                    authoritative_metadata=self._authoritative_metadata(item,documents)
                    if authoritative_text!=text or authoritative_metadata!=dict(metadata or {}):
                        metadata_changed=True
                    kept.append((vector,authoritative_text,authoritative_metadata,key));seen.add(key)
        changed=fingerprint_changed or index_shape_changed or metadata_changed or len(kept)!=len(self._ids)
        missing=[item for key,item in required.items() if key not in seen]
        if missing:
            texts=[str(item.get("text") or "") for item in missing]
            vectors=self.embeddings.embed_documents(texts)
            kept.extend(
                (
                    np.array(vector,dtype=np.float32),text,
                    self._authoritative_metadata(item,documents),
                    str(item.get("chunk_id")),
                )
                for item,text,vector in zip(missing,texts,vectors)
            )
            changed=True
        if changed:
            if kept:
                self._vecs,self._texts,self._metas,self._ids=map(list,zip(*kept))
            else:
                self._vecs=[];self._texts=[];self._metas=[];self._ids=[]
            self._save()
        return {
            "changed":changed,
            "chunks":len(self._ids),
            "recreated":len(missing),
            "fingerprint_changed":fingerprint_changed,
            "identity":"chunk_id",
        }

    def repair_from_storage(self):
        """Explicit maintenance entry point for rebuilding the global index."""
        return self.reconcile_with_storage()

    @_synchronized
    def consistency_status(self):
        """Compare the derived index shape and identities with authoritative SQLite.

        This is intentionally read-only and cheap enough for startup. Projection
        state rows describe the last successful write; they cannot prove that a
        derived file still exists or has not been truncated afterwards.
        """
        self._load()
        from backend.storage import get_repository

        required_ids = {
            str(item.get("chunk_id"))
            for item in get_repository().document_chunks()
            if item.get("chunk_id")
        }
        indexed_ids = {str(item) for item in self._ids if str(item)}
        shape_valid = len(self._vecs) == len(self._texts) == len(self._metas) == len(self._ids)
        fingerprint_valid = self._saved_index_fingerprint == self._current_index_fingerprint()
        missing = required_ids - indexed_ids
        stale = indexed_ids - required_ids
        return {
            "consistent": bool(shape_valid and fingerprint_valid and not missing and not stale),
            "required_chunks": len(required_ids),
            "indexed_chunks": len(indexed_ids),
            "missing_chunks": len(missing),
            "stale_chunks": len(stale),
            "shape_valid": shape_valid,
            "fingerprint_valid": fingerprint_valid,
        }

    def _active_indexes(self, indexes):
        """Apply an authoritative lifecycle check immediately before returning results."""
        candidates=[int(index) for index in indexes if 0<=int(index)<len(self._ids)]
        if not candidates:
            return []
        from backend.storage import get_repository

        active=get_repository().active_chunk_ids([self._ids[index] for index in candidates])
        return [index for index in candidates if str(self._ids[index]) in active]

    def _authoritative_index_metadata(self, indexes):
        """Return terminally authorized indexes with current SQLite metadata."""
        candidates=[int(index) for index in indexes if 0<=int(index)<len(self._ids)]
        if not candidates:
            return []
        from backend.storage import get_repository

        repository=get_repository()
        active=repository.active_chunk_ids([self._ids[index] for index in candidates])
        documents={
            str(item.get("document_id")):dict(item)
            for item in repository.list_documents()
            if isinstance(item,dict) and item.get("document_id")
        }
        by_alias: Dict[str,List[Dict[str,Any]]]={}
        for document in documents.values():
            for alias in (document.get("stored_file"),document.get("source_file")):
                alias=str(alias or "").strip()
                if alias:
                    by_alias.setdefault(alias,[]).append(document)
        result=[]
        for index in candidates:
            if str(self._ids[index]) not in active:
                continue
            stored_metadata=dict(self._metas[index] or {})
            document_id=str(stored_metadata.get("document_id") or "")
            if document_id not in documents:
                aliases=[
                    str(stored_metadata.get("stored_file") or "").strip(),
                    str(stored_metadata.get("source_file") or "").strip(),
                ]
                matches={
                    str(document.get("document_id")):document
                    for alias in aliases for document in by_alias.get(alias,[])
                }
                if len(matches)!=1:
                    continue
                document_id,parent=next(iter(matches.items()))
                documents={**documents,document_id:parent}
            result.append((index,self._authoritative_metadata({
                "chunk_id":str(self._ids[index]),
                "document_id":document_id,
                "chunk_index":stored_metadata.get("chunk_index"),
                "metadata":stored_metadata,
            },documents)))
        return result

    def _documents_for_indexes(self, indexes):
        """Materialize results only after a final authoritative lifecycle check."""
        return [
            Document(page_content=self._texts[index],metadata=metadata)
            for index,metadata in self._authoritative_index_metadata(indexes)
        ]
    @_synchronized
    def reinitialize(self):
        self._embeddings=None
        return True
    @_synchronized
    def count(self):
        self._load()
        return len(self._active_indexes(range(len(self._ids))))
    @_synchronized
    def get(self, include_vectors: bool = False):
        """Return the current snapshot after an authoritative terminal filter."""
        self._load()
        snapshot=self._authoritative_index_metadata(range(len(self._ids)))
        indexes=[index for index,_ in snapshot]
        data = {
            "ids":[self._ids[index] for index in indexes],
            "documents":[self._texts[index] for index in indexes],
            "metadatas":[metadata for _,metadata in snapshot],
        }
        if include_vectors:
            data["vectors"] = [self._vecs[index].tolist() for index in indexes]
        return data

    @_synchronized
    def similarity_search(self,query,k=None,score_threshold=None):
        k=k or settings.RETRIEVER_K
        threshold=settings.RETRIEVER_SCORE_THRESHOLD if score_threshold is None else score_threshold
        self._load()
        if not self._vecs:
            return[]
        q_vec=np.array(self.embeddings.embed_query(query),dtype=np.float32)
        matrix=np.array(self._vecs)
        q_norm=np.linalg.norm(q_vec)
        if q_norm<1e-10:
            return[]
        scores=np.dot(matrix,q_vec)/(np.linalg.norm(matrix,axis=1)*q_norm+1e-10)
        active_indexes=self._active_indexes(range(len(self._ids)))
        top_idx=sorted(active_indexes,key=lambda index:float(scores[index]),reverse=True)[:k]
        eligible=[idx for idx in top_idx if float(scores[idx])>=threshold]
        return self._documents_for_indexes(eligible)
    @_synchronized
    def similarity_search_with_relevance_scores(self,query,k=None,score_threshold=None):
        k=k or settings.RETRIEVER_K
        threshold=settings.RETRIEVER_SCORE_THRESHOLD if score_threshold is None else score_threshold
        self._load()
        if not self._vecs:return[]
        q_vec=np.array(self.embeddings.embed_query(query),dtype=np.float32)
        matrix=np.array(self._vecs)
        q_norm=np.linalg.norm(q_vec)
        if q_norm<1e-10:return[]
        scores=np.dot(matrix,q_vec)/(np.linalg.norm(matrix,axis=1)*q_norm+1e-10)
        active_indexes=self._active_indexes(range(len(self._ids)))
        top_idx=[
            index for index in sorted(active_indexes,key=lambda index:float(scores[index]),reverse=True)
            if float(scores[index])>=threshold
        ][:k]
        final_snapshot=self._authoritative_index_metadata(top_idx)
        return[(Document(page_content=self._texts[i],metadata=metadata),float(scores[i]))for i,metadata in final_snapshot]

    def _lexical_tokens(self, text):
        text = _normalize_text(text).lower()
        tokens = []
        for word in TEXT_EN.findall(text):
            if len(word) > 1:
                tokens.append(word)
        for sequence in TEXT_CN.findall(text):
            for char in sequence:
                tokens.append(char)
            for size in (2, 3, 4):
                for index in range(max(0, len(sequence) - size + 1)):
                    tokens.append(sequence[index:index + size])
        for number in TEXT_NUM.findall(text):
            tokens.append(number)
        return tokens

    @_synchronized
    def hybrid_search_with_relevance_scores(self, query, k=None, fetch_k=None, score_threshold=None):
        """Apply a raw-vector evidence gate before lexical reranking."""
        k = k or settings.RETRIEVER_K
        threshold=settings.RETRIEVER_SCORE_THRESHOLD if score_threshold is None else score_threshold
        self._load()
        if not self._vecs:
            return []

        query_vector = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        matrix = np.array(self._vecs)
        query_norm = np.linalg.norm(query_vector)
        if query_norm < 1e-10:
            return []
        vector_scores = np.dot(matrix, query_vector) / (np.linalg.norm(matrix, axis=1) * query_norm + 1e-10)
        active_indexes = self._active_indexes(range(len(self._ids)))
        limit = min(fetch_k or max(k * 5, 12), len(active_indexes))
        raw_gate=max(0.0,float(threshold))
        candidate_indexes = [
            index for index in sorted(
                active_indexes,key=lambda index:float(vector_scores[index]),reverse=True
            )
            if float(vector_scores[index])>raw_gate
        ][:limit]
        authoritative_metadata=dict(self._authoritative_index_metadata(candidate_indexes))
        candidate_indexes=[index for index in candidate_indexes if index in authoritative_metadata]
        query_tokens = set(self._lexical_tokens(query))
        query_text = (query or "").lower()
        ranked = []

        for index in candidate_indexes:
            metadata = authoritative_metadata[index]
            searchable = " ".join([
                self._texts[index] or "",
                str(metadata.get("source_file") or ""),
                str(metadata.get("category") or ""),
                " ".join(str(item) for item in metadata.get("categories", []) if item),
                str(metadata.get("role") or ""),
            ])
            document_tokens = set(self._lexical_tokens(searchable))
            overlap = query_tokens & document_tokens
            if query_tokens:
                weighted_overlap = sum(2 if len(token) >= 2 else 0.35 for token in overlap)
                lexical_score = min(1.0, weighted_overlap / max(1.0, len(query_tokens) * 0.6))
            else:
                lexical_score = 0.0
            phrase_bonus = 0.0
            for phrase in TEXT_CN.findall(query_text):
                if len(phrase) >= 2 and phrase in searchable.lower():
                    phrase_bonus += 0.12
            normalized_vector = max(0.0, min(1.0, (float(vector_scores[index]) + 1.0) / 2.0))
            score = normalized_vector * 0.58 + lexical_score * 0.34 + min(0.18, phrase_bonus)
            ranked.append((index,float(score)))

        ranked.sort(key=lambda item: item[1], reverse=True)
        final_snapshot=dict(self._authoritative_index_metadata([item[0] for item in ranked[:k]]))
        return [
            (Document(page_content=self._texts[index],metadata=final_snapshot[index]),score)
            for index,score in ranked[:k] if index in final_snapshot
        ]

    @_synchronized
    def list_documents(self):
        from backend.storage import get_repository

        return get_repository().list_documents()
    @_synchronized
    def delete_document(self,filename):
        self.reconcile_with_storage()
        before=len(self._ids)
        from backend.storage import get_repository

        repository=get_repository()
        snapshots=repository.delete_document_with_snapshot(filename,all_versions=False)
        try:
            self.reconcile_with_storage()
        except Exception:
            repository.restore_document_snapshots(snapshots)
            try:
                self.reconcile_with_storage()
            except Exception:
                logger.exception("restore derived index after compensated document deletion failed")
            raise
        return before-len(self._ids)

    @_synchronized
    def delete_documents_by_source(self,filename):
        self.reconcile_with_storage()
        before=len(self._ids)
        from backend.storage import get_repository

        repository=get_repository()
        snapshots=repository.delete_document_with_snapshot(filename,all_versions=True)
        try:
            self.reconcile_with_storage()
        except Exception:
            repository.restore_document_snapshots(snapshots)
            try:
                self.reconcile_with_storage()
            except Exception:
                logger.exception("restore derived index after compensated source deletion failed")
            raise
        return before-len(self._ids)

    @_synchronized
    def max_marginal_relevance_search(self,query,k=4,fetch_k=8,lambda_mult=0.5,score_threshold=None):
        self._load()
        if not self._vecs:
            return[]
        k=min(k,len(self._vecs))
        fetch_k=min(fetch_k,len(self._vecs))
        q_vec=np.array(self.embeddings.embed_query(query),dtype=np.float32)
        matrix=np.array(self._vecs)
        q_norm=np.linalg.norm(q_vec)
        if q_norm<1e-10:
            return[]
        scores=np.dot(matrix,q_vec)/(np.linalg.norm(matrix,axis=1)*q_norm+1e-10)
        threshold=settings.RETRIEVER_SCORE_THRESHOLD if score_threshold is None else score_threshold
        active_indexes=self._active_indexes(range(len(self._ids)))
        top_idx=[
            index for index in sorted(active_indexes,key=lambda index:float(scores[index]),reverse=True)
            if float(scores[index])>=threshold
        ][:fetch_k]
        if not top_idx:
            return[]
        if len(top_idx)<=k:
            return self._documents_for_indexes(top_idx)
        selected=[]
        remaining=list(top_idx)
        selected.append(remaining.pop(0))
        while len(selected)<k and remaining:
            best_idx=None
            best_mmr=-float("inf")
            for r_idx in remaining:
                sim_q=float(scores[r_idx])
                max_sim_s=max(float(np.dot(matrix[r_idx],matrix[s_idx])/(np.linalg.norm(matrix[r_idx])*np.linalg.norm(matrix[s_idx])+1e-10))for s_idx in selected)
                mmr=lambda_mult*sim_q-(1-lambda_mult)*max_sim_s
                if mmr>best_mmr:
                    best_mmr=mmr
                    best_idx=r_idx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        return self._documents_for_indexes(selected)


vector_store=VectorStore()
