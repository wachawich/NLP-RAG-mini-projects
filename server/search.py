from typing import List, Dict
import json


def llm_rewrite_query(question: str, groq_client) -> str:
    system_prompt = (
        "You rewrite user queries to be clearer and more suitable for document retrieval. "
        "Keep the meaning the same, but remove noise and make it concise."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def get_all_namespaces(index) -> List[str]:
    stats = index.describe_index_stats()
    namespaces_dict = stats.get("namespaces") or {}
    namespaces = list(namespaces_dict.keys())
    return namespaces


def llm_route_namespaces(
    groq_client,
    rewritten_query: str,
    all_namespaces: List[str],
    top_k: int = 4
) -> List[str]:
    if not all_namespaces:
        return []

    system_prompt = (
        "You are a routing model. Your job is to choose the most relevant namespaces "
        f"from this list: {all_namespaces}. "
        "Given a query, you MUST return ONLY a JSON array of strings, each string "
        f"{top_k} namespaces, sorted from most to least relevant. "
        "Do not add any explanation or extra keys."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rewritten_query},
    ]

    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        routed = [ns for ns in parsed if ns in all_namespaces]
    except Exception:
        routed = []
    return routed[:top_k]

def search_in_namespaces(
    model,
    index,
    query_text: str,
    namespaces: List[str],
    per_namespace: int = 5
) -> List[Dict]:
    """
    ยิง query เข้า Pinecone หลาย namespace
    - encode query แค่ครั้งเดียว
    - loop ผ่านแต่ละ namespace
    - รวมผลเป็น list เดียว

    return: list ของ dict ที่เก็บข้อมูล id, namespace, score, text, metadata
    """
    if not namespaces:
        return []

    results: List[Dict] = []

    # 1) ทำ embedding
    q_emb = model.encode(query_text, normalize_embeddings=True).tolist()

    # 2) ยิง query เข้า Pinecone ในแต่ละ namespace
    for ns in namespaces:
        res = index.query(
            vector=q_emb,
            top_k=per_namespace,
            include_metadata=True,
            namespace=ns,
        )

        # 3) เก็บ match แต่ละตัว
        for match in res.matches:
            results.append({
                "id": match.id,
                "namespace": ns,
                "score": float(match.score),
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
            })

    return results


def cohere_rerank(co, query: str, passages: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Rerank ด้วย Cohere Rerank v3.5
    """
    if not passages:
        return []

    docs = [p["text"] for p in passages]

    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=top_k,
    )

    ranked = []
    for r in response.results:
        ranked.append(passages[r.index])

    return ranked

def rag_search(
    groq_client,
    model,
    index,
    co,
    question: str,
    ns_top_k: int = 4,
    per_namespace: int = 5,
    final_k: int = 10,
) -> Dict:
    """
    ทั้ง flow:
      1) รับคำถามดิบ (question)
      2) ใช้ LLM rewrite query
      3) ใช้ LLM routing เลือก namespace (สูงสุด ns_top_k อัน)
      4) ใช้ embedding search ในแต่ละ namespace (per_namespace ต่ออัน)
      5) รวม candidates แล้ว rerank เหลือ final_k
    """
    # 1) rewrite query
    rewritten = llm_rewrite_query(question, groq_client)

    # 2) namespace routing
    all_ns = get_all_namespaces(index)
    routed_ns = llm_route_namespaces(groq_client, rewritten, all_ns, top_k=ns_top_k)

    # ถ้า LLM ไม่เลือกอะไรเลย ใช้ namespace ทั้งหมด
    if not routed_ns:
        routed_ns = all_ns

    # 3) multi-namespace search
    passages = search_in_namespaces(
        model,
        index,
        query_text=rewritten,
        namespaces=routed_ns,
        per_namespace=per_namespace,
    )

    # 4) rerank → top_k
    top_passages = cohere_rerank(co, rewritten, passages, top_k=final_k)

    # return {
    #     "original_question": question,
    #     "rewritten_query": rewritten,
    #     "namespaces_used": routed_ns,
    #     "candidate_count": len(passages),
    #     "top_passages": top_passages,
    # }

    # ถ้าอยากเอาแค่ list text เอาคอมเม้นออก
    return [p["text"] for p in top_passages]
