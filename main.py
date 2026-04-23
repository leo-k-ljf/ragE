import json
import os
from pathlib import Path
from typing import List, Tuple, Any, Literal


# 通过段落切片
def split_text(doc: str) -> List[str]:
    with open(doc, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.split('\n\n')


# 固定窗口切片
def split_text_fixed(text: str, chunk_size: int = 300) -> List[str]:
    if not text:
        return []

    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')


def embedding_text(texts: List[str]) -> List[Tuple[str, List[float]]]:
    if not texts:
        return []

    # 批量生成向量
    embeddings = embedding_model.encode(texts)

    # 返回文本和向量的配对
    return [(text, embedding.tolist()) for text, embedding in zip(texts, embeddings)]


# 向量数据库
import chromadb

client = chromadb.PersistentClient(path="./my_chromadb_data")
collection = client.get_or_create_collection(name="my_collection")

def add_vector_to_db(
    vectors: List[Tuple[str, List[float]]],
    ids: List[str] = None,
    metadatas: List[dict] = None
):
    if not vectors:
        return

    if ids is None:
        ids = [f"vec_{i}" for i in range(len(vectors))]
    documents = [text for text, _ in vectors]
    embeddings = [embedding for _, embedding in vectors]

    kwargs = {
        "ids": ids,
        "documents": documents,
        "embeddings": embeddings
    }
    if metadatas is not None:
        kwargs["metadatas"] = metadatas

    collection.add(**kwargs)

    print(f"成功添加 {len(vectors)} 个向量到数据库")


def extract_text_from_item(item: dict) -> str:
    """从 JSON 元素中提取用于 embedding 的文本"""
    parts = []
    if item.get("name"):
        parts.append(str(item["name"]))
    if item.get("description"):
        parts.append(str(item["description"]))
    return "\n".join(parts)


def load_and_embed_json_files(directory: str):
    """
    读取目录下所有 JSON 文件，对其数组元素进行 embedding 并存入数据库。
    每个元素需要包含可用于唯一标识的 id 或 name 字段。
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        print(f"目录下没有找到 JSON 文件: {directory}")
        return

    all_texts = []
    all_ids = []
    all_metadatas = []

    for file_path in json_files:
        print(f"正在读取: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  跳过（JSON 解析失败）: {file_path.name} - {e}")
                continue

        if not isinstance(data, list):
            print(f"  跳过（根元素不是数组）: {file_path.name}")
            continue

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            text = extract_text_from_item(item)
            if not text.strip():
                continue

            # 构造唯一 ID: 文件名_索引_元素的 id 或 name
            item_id = item.get("id") or item.get("name") or f"item_{idx}"
            vec_id = f"{file_path.stem}_{idx}_{item_id}"

            all_texts.append(text)
            all_ids.append(vec_id)
            all_metadatas.append({
                "source_file": file_path.name,
                "index": idx,
                "raw_json": json.dumps(item, ensure_ascii=False)
            })

    if not all_texts:
        print("没有可嵌入的数据")
        return

    print(f"共读取 {len(all_texts)} 条数据，开始生成向量...")
    vectors = embedding_text(all_texts)
    add_vector_to_db(vectors, ids=all_ids, metadatas=all_metadatas)
    print("全部完成！")


def search_similar(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results




from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")

def rerank_text(query: str, texts: List[str]) -> List[Tuple[float, str]]:
    if not texts:
        return []

    rankings = cross_encoder.rank(query, texts)
    return [(item["score"], texts[item["corpus_id"]]) for item in rankings]


# texts = ["你好世界", "人工智能", "机器学习", "爪击", "冷静头脑"]
# vectors = embedding_text(texts)
# add_vector_to_db(vectors)
# print(f"生成了{len(vectors)}个向量")

if __name__ == '__main__':
    # load_and_embed_json_files(r"D:\spire-codex\data\zhs")
    # 召回
    similar = search_similar("Act 1 Boss", top_k=10)
    docs = similar.get('documents')
    print(f"similar:{docs}")
    wait_to_rerank = docs[0] if docs else []
    # 重排
    rr = rerank_text("Act 1 Boss", wait_to_rerank)
    for score, text in rr:
        print(f"score: {score:.4f}, text: {text}")