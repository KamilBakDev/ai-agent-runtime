# Architecture

This document explains the *why* behind the structure in [`README.md`](../README.md) -- the
tradeoffs made to keep this genuinely runnable and testable offline while still exercising real
production patterns.

## 1. State and checkpointing

The orchestrator (`agents/orchestrator.py`) is a `langgraph.graph.StateGraph` over a single
`AgentState` TypedDict (`messages`, `research_notes`, `code_draft`, `review_status`,
`revision_count`, `final_output`, ...). Three nodes: `researcher -> coder -> reviewer -> END`.

**Why LangGraph owns the checkpoint tables, not a custom schema.** `AsyncPostgresSaver`
(`langgraph-checkpoint-postgres`) persists a full snapshot of `AgentState` after every node, keyed
by `(thread_id, checkpoint_id)`. `thread_id` is set to our `Session.id`, so a session *is* a
LangGraph thread. Building a parallel `checkpoints` table in `db/models.py` would mean keeping two
sources of truth in sync for no benefit -- `agents/service.py` reads checkpoint history straight
from the saver via `graph.aget_state_history()`.

**Time-travel and branching** (`AgentService.restore_checkpoint` /
`AgentService.branch_from_checkpoint` in `agents/service.py`) both lean on the same mechanism:
`{"configurable": {"thread_id": ..., "checkpoint_id": ...}}` addresses any historical checkpoint.
- *Restore* re-applies an empty update at that checkpoint (attributing it to the node that
  produced it, recovered from the checkpoint's own metadata) so it becomes the thread's new head
  -- non-destructively; older "future" checkpoints stay in history.
- *Branch* reads that checkpoint's values and seeds a **new** thread (`aupdate_state` on a
  thread with no prior checkpoint initializes it), giving a genuinely separate session with its
  own future.

## 2. Human-in-the-loop

`compile_graph()` compiles with `interrupt_before=["reviewer"]` -- a *static* interrupt, chosen
over LangGraph's newer dynamic `interrupt()` call because it maps directly onto "pause before this
node, resume when a human writes a decision into state," which is exactly what the API needs, and
it's trivial to test (`graph.ainvoke(...)` returns having paused; `graph.aupdate_state(...)` +
`graph.ainvoke(None, ...)` resumes).

`agents/subagents/reviewer.py::route_after_review` is the branching logic: `rejected` loops back
to `coder` (bounded by `MAX_REVISIONS = 3` so a stuck reviewer can't infinite-loop the graph),
`approved`/`edited` finish. This is the graph's one piece of real conditional branching, per the
brief's "add branching" requirement -- it was more valuable to make the reject/revise loop genuinely
work end-to-end (with a real bound) than to add a second, cosmetic branch elsewhere.

## 3. RAG: why Qdrant-only, why a fake embeddings provider by default

The brief allowed "Qdrant or Chroma." Qdrant was chosen because the acceptance criteria
(`docker-compose up` standing up Postgres + Qdrant + API) already commits to it, and maintaining
two vector backends would add maintenance surface without adding anything to demonstrate.

**Embeddings are pluggable** (`rag/embeddings.py`): `fake` is a deterministic hash-bucket
embedding -- zero dependencies, zero network, stable across runs -- and is the default so RAG is
testable exactly the same way the LLM is. `ollama`/`openai` are real embedding calls, wired but
not exercised in CI. The **retriever** (`rag/retriever.py`) fuses Qdrant's cosine ranking with a
BM25 lexical ranking (`rank_bm25`) via reciprocal rank fusion -- with the fake embedding's cosine
signal being fairly weak on its own, BM25 is what makes retrieval actually work well offline (see
`rag/evals/evaluate.py`: 100% retrieval-hit-rate and keyword-faithfulness on the 6-question eval
set, LLM-free by design so it costs nothing to run in CI).

## 4. MCP: a real subprocess, not a shim

`agents/tools/mcp_server.py` is a real `mcp` SDK server (note: this repo pins `mcp>=1.0.0` but
resolves to the 2.x line, where `FastMCP` was renamed `MCPServer` -- see the file for the exact
import). `agents/tools/mcp_client.py` spawns it as a stdio subprocess and talks the actual MCP
protocol. This matters for two reasons: (1) it's what "agents call tools through an MCP client"
is supposed to mean, not a relabeled function call, and (2) it caught a real protocol-shape bug --
this SDK serializes a `list[dict]` tool return as one text content block *per list item*, not one
JSON array, which silently produced wrong results (reading a single chunk's dict keys as if they
were a result count) until the client was fixed to re-collect every block. `tests/test_mcp.py`
exercises the actual subprocess round trip, not a mock.

## 5. A2A: independent HTTP services, not renamed function calls

`api/routes/a2a.py`'s `/a2a/invoke` reimplements each agent's logic as a **stateless** HTTP
handler, separate from the LangGraph orchestrator's in-process nodes. `agents/tools/a2a_client.py`
looks the target agent up in `A2A_REGISTRY` (`{name: base_url}`) and makes a real HTTP POST.
`tests/test_a2a.py` proves this over `httpx.ASGITransport` (a real HTTP request/response cycle,
just without an actual socket) rather than calling the handler function directly -- the point of
A2A is that agents are independently *addressable*, which a bare function call can't prove.

## 6. Observability

OpenTelemetry (`api/middleware/tracing.py`) wraps every request in a span (`console` exporter by
default -- zero infra to see traces locally; set `OTEL_EXPORTER_OTLP_ENDPOINT` to ship to
Jaeger/Tempo). LangSmith tracing is opt-in: `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` set
the env vars LangChain itself reads; absent, nothing is touched and nothing can fail.
`agents/observability.py` adds a small in-process metrics store (request latency
p50/p95 over the last 1000 requests, per-node LLM token usage extracted from
`AIMessage.usage_metadata`/`response_metadata` where the provider reports it) behind `GET
/metrics`. This is deliberately not a Prometheus/OTel-metrics pipeline -- for a local demo, a
plain JSON snapshot is more useful than standing up a metrics stack, and the shape mirrors what a
real exporter would report if this were promoted to one.

## 7. What's intentionally out of scope

- **Kubernetes**: `infra/docker-compose.yml` is the one deployment target that matters for "clone
  and run this." Writing k8s manifests nobody will `kubectl apply` doesn't demonstrate anything
  `docker-compose.yml` doesn't already.
- **GHCR push, not a live deployment**: `cd.yml` builds and pushes an image; there's no cloud
  target to deploy it to for a portfolio repo, and standing one up would cost money for no
  additional signal.
- **Cost-per-request**: token usage is tracked (`/metrics`); converting it to a dollar figure needs
  a maintained per-model price table, which is a data-freshness liability more than an
  architecture decision -- left as a natural extension of the token-usage data that's already
  captured.
- **LLM-as-judge RAG evaluation**: `rag/evals/evaluate.py` uses retrieval-hit-rate and
  keyword-overlap instead, specifically so the eval suite runs in CI with no LLM calls. Swapping in
  an LLM judge is a small addition once a real provider is configured.
