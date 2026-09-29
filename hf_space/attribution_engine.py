"""
Sentence-Level Citation & Hallucination Guardrail Attribution Engine
Performs granular sentence-to-chunk alignment to verify groundedness,
detect unverified claims / hallucinations, and inject interactive citations.
"""

import re
from typing import List, Dict, Any

def split_sentences(text: str) -> List[str]:
    """Splits text into discrete, substantive sentences while ignoring formatting artifacts."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = []
    for s in raw:
        clean = s.strip()
        # Filter trivial headings or punctuation lines
        if len(clean) > 15 and not clean.startswith(('#', '*', '-', '|')):
            sentences.append(clean)
    return sentences

def compute_sentence_overlap(sentence: str, chunk: str) -> float:
    """Computes lexical-semantic token containment between a claim and an evidence chunk."""
    s_tokens = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
    c_tokens = set(re.findall(r'\b\w{3,}\b', chunk.lower()))
    if not s_tokens:
        return 0.0
    overlap = len(s_tokens.intersection(c_tokens))
    return overlap / len(s_tokens)

def attribute_citations(generation: str, documents: List[Any]) -> Dict[str, Any]:
    """
    Deconstructs generation into claims, aligns them with source chunks,
    and returns a structured attribution matrix.
    """
    if not generation:
        return {
            "attribution_score": 1.0,
            "grounded_ratio": "100%",
            "claims": [],
            "annotated_html": ""
        }

    if not documents:
        return {
            "attribution_score": 0.0,
            "grounded_ratio": "0%",
            "claims": [],
            "annotated_html": generation
        }

    chunks = [d.page_content if hasattr(d, 'page_content') else str(d) for d in documents]
    sentences = split_sentences(generation)
    
    if not sentences:
        return {
            "attribution_score": 1.0,
            "grounded_ratio": "100%",
            "claims": [],
            "annotated_html": generation
        }

    claims_data = []
    grounded_count = 0
    annotated_sentences = []

    for idx, s in enumerate(sentences):
        best_chunk_idx = None
        best_score = 0.0

        for c_idx, chunk in enumerate(chunks):
            score = compute_sentence_overlap(s, chunk)
            if score > best_score:
                best_score = score
                best_chunk_idx = c_idx

        # Grounding threshold: at least 35% token containment in technical domain chunks
        is_grounded = best_score >= 0.35 and best_chunk_idx is not None
        if is_grounded:
            grounded_count += 1
            citation_num = best_chunk_idx + 1
            annotated_s = f"{s} <sup class='citation-tag' data-chunk='{best_chunk_idx}' title='Verified by Grounded Evidence Chunk [{citation_num}] (Confidence: {round(best_score*100)}%)'>[{citation_num}]</sup>"
        else:
            annotated_s = f"<span class='unverified-claim' title='Unsubstantiated claim: Evidence chunk support < 35%'>{s}</span>"

        annotated_sentences.append(annotated_s)
        claims_data.append({
            "claim_index": idx + 1,
            "sentence": s,
            "grounded": is_grounded,
            "confidence": round(best_score, 3),
            "cited_chunk_index": best_chunk_idx + 1 if is_grounded else None
        })

    attribution_score = round(grounded_count / max(len(sentences), 1), 3)

    return {
        "attribution_score": attribution_score,
        "grounded_ratio": f"{int(attribution_score * 100)}%",
        "total_claims": len(sentences),
        "grounded_claims": grounded_count,
        "claims": claims_data,
        "annotated_text": " ".join(annotated_sentences)
    }
