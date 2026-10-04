"""LLM factory + answer generation with citation validation."""
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore

from rag_app.config import Settings
from rag_app.generation.prompts import QA_PROMPT
from rag_app.generation.schemas import Answer, Claim
from rag_app.logging_utils import get_logger

log = get_logger(__name__)


def build_llm(s: Settings) -> LLM:
    if s.llm_provider == "openai":
        from llama_index.llms.openai import OpenAI

        # api_key from settings (.env/env); if None, OpenAI() falls back to OPENAI_API_KEY in the env.
        return OpenAI(model=s.llm_model, temperature=s.llm_temperature, api_key=s.openai_api_key)
    if s.llm_provider == "vllm":
        # vLLM exposes an OpenAI-compatible API: `vllm serve <model>`
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=s.llm_model,
            api_base=s.llm_api_base,
            api_key="EMPTY",
            is_chat_model=True,
            temperature=s.llm_temperature,
        )
    raise ValueError(f"Unknown llm_provider: {s.llm_provider}")


def build_source_map(nodes: list[NodeWithScore]) -> dict[str, NodeWithScore]:
    """Short ids (S1, S2...) are easier for the LLM to cite than long UUIDs."""
    return {f"S{i + 1}": n for i, n in enumerate(nodes)}


def format_context(source_map: dict[str, NodeWithScore]) -> str:
    blocks = []
    for sid, n in source_map.items():
        meta = n.node.metadata
        label = meta.get("file_name") or meta.get("source_url", "unknown")
        page = f", page {meta['page_label']}" if "page_label" in meta else ""
        blocks.append(f"[{sid}] ({label}{page})\n{n.node.get_content()}")
    return "\n\n".join(blocks)


def validate_citations(answer: Answer, valid_ids: set[str]) -> Answer:
    """Drop claims that cite nothing real. (Step up: also check the source SUPPORTS the claim.)"""
    kept: list[Claim] = []
    for c in answer.claims:
        good = [sid for sid in c.source_ids if sid in valid_ids]
        if good:
            kept.append(Claim(text=c.text, source_ids=good))
        else:
            log.warning("Dropped uncited claim: %s", c.text)
    insufficient = answer.insufficient_context or not kept
    return Answer(
        summary="" if insufficient else answer.summary,
        claims=kept,
        insufficient_context=insufficient,
    )


def repair_citations(embed_model, answer: Answer, source_map: dict[str, NodeWithScore],
                     min_sim: float = 0.55) -> Answer:
    """Re-ground each claim's citations to the context chunk(s) that actually support it.

    LLMs (especially small ones) often attach a plausible-but-wrong source id. We embed
    each claim and each context chunk and reassign the citation to the most similar
    chunk(s), so 'cites its source' is true rather than decorative.
    """
    if not answer.claims or not source_map:
        return answer
    import numpy as np

    sids = list(source_map)
    mat = np.array(embed_model.get_text_embedding_batch(
        [source_map[s].node.get_content() for s in sids]))
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    repaired: list[Claim] = []
    for c in answer.claims:
        cv = np.array(embed_model.get_text_embedding(c.text))
        cv /= (np.linalg.norm(cv) + 1e-9)
        sims = mat @ cv
        order = list(np.argsort(-sims))
        strong = [sids[i] for i in order if float(sims[i]) >= min_sim]
        chosen = strong[:2] if strong else [sids[order[0]]]  # fall back to the closest chunk
        repaired.append(Claim(text=c.text, source_ids=chosen))
    return Answer(summary=answer.summary, claims=repaired,
                  insufficient_context=answer.insufficient_context)


def generate_answer(llm: LLM, question: str, nodes: list[NodeWithScore],
                    embed_model=None) -> tuple[Answer, dict[str, NodeWithScore]]:
    source_map = build_source_map(nodes)
    if not source_map:
        return Answer(insufficient_context=True), source_map
    raw = llm.structured_predict(Answer, QA_PROMPT, question=question, context=format_context(source_map))
    answer = validate_citations(raw, set(source_map))
    if embed_model is not None:
        answer = repair_citations(embed_model, answer, source_map)
    return answer, source_map
