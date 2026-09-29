---
title: Aura AI Adaptive RAG Backend
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Aura AI — Adaptive Multi-Strategy RAG Research Backend

Full Python backend powered by LangGraph, ChromaDB, NetworkX, FlashRank Neural Cross-Encoder, and Sentence-Level Claim Attribution.

## API Endpoints:
- `GET /api/graph` — Full 70-node entity-relationship graph
- `GET /api/graph/path?source=...&target=...` — Multi-hop relational shortest path tracer
- `POST /api/chat` — Adaptive query routing with sentence-level claim citations [1], [2]
- `POST /api/arena` — Concurrent 4-way Strategy Arena evaluation
- `POST /api/upload` — Real-time document parser & ChromaDB vector indexer
