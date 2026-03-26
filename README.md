# O2C Graph Intelligence — SAP Order to Cash

A context graph system with an LLM-powered natural language query interface for exploring SAP Order-to-Cash (O2C) business process data.

---

## Table of Contents

- [Overview](#overview)
- [Architecture Decisions](#architecture-decisions)
- [Database Choice](#database-choice)
- [Graph Modeling](#graph-modeling)
- [LLM Integration & Prompting Strategy](#llm-integration--prompting-strategy)
- [Guardrails](#guardrails)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [Example Queries](#example-queries)
- [Tech Stack](#tech-stack)

---

## Overview

This system ingests fragmented SAP O2C data (orders, deliveries, invoices, payments, customers, products) and unifies it into a directed graph. Users interact with the graph via a chat interface — asking questions in plain English — and the system translates those into structured graph queries, executes them, and returns data-backed answers visualized on the graph canvas.

```
User Query (natural language)
        ↓
   LLM (Groq / llama-3.1-8b-instant)
        ↓
   Intent Classification → Structured Query
        ↓
   NetworkX Graph Traversal
        ↓
   JSON Response → vis-network Visualization + Chat Answer
```

---

## Architecture Decisions

### 1. Backend: FastAPI (Python)

FastAPI was chosen over Flask or Django for three specific reasons:

- **Async-ready**: O2C datasets can be large. FastAPI's async support means the server handles concurrent requests without blocking while graph queries run.
- **Auto-validation**: Pydantic models catch bad inputs before they reach graph logic, which is important when user queries might produce unpredictable LLM outputs.
- **Speed**: FastAPI is one of the fastest Python web frameworks available, which matters when the graph traversal and LLM call happen in sequence per request.

### 2. Graph Library: NetworkX (DiGraph)

NetworkX's `DiGraph` (Directed Graph) was chosen because the O2C flow is fundamentally directional:

```
Sales Order → Delivery → Billing Document → Journal Entry
```

A directed graph correctly models this — a delivery does not "point back" to an order in business logic. NetworkX provides:

- Path-finding (`nx.shortest_path`, `nx.all_simple_paths`)
- Neighbor traversal for node expansion
- Cycle detection for identifying broken flows
- In-memory speed for datasets of this scale

### 3. Frontend: Vanilla HTML + vis-network

No React or Vue framework was used intentionally. The reasons:

- **Zero build step**: The frontend is a single `index.html` that works by opening in a browser. This removes all npm/webpack complexity from the demo setup.
- **vis-network**: Industry-standard graph visualization library, stable, well-documented, and handles physics simulation, node dragging, hover, and click events natively.
- **Portability**: Any evaluator can open the file without installing anything beyond running the Python backend.

### 4. Request Flow

Every user query follows this exact pipeline:

```
1. User types query in chat input
2. Frontend POSTs to GET /query?q=<encoded_query>
3. FastAPI receives query, calls build_graph() if not yet loaded
4. Query is passed to ask_llm() → Groq API returns intent classification
5. Intent is used to select graph traversal logic
6. Results returned as JSON {intent, data}
7. Frontend renders nodes/edges in vis-network + displays chat bubble
```

### 5. Graph Loading Strategy: Lazy Singleton

The graph is built once on the first API request and cached in memory (via the `DATA_LOADED` flag and global `G`). This avoids:

- Re-parsing JSONL files on every request
- Rebuilding node/edge indices on every query
- Unnecessary disk I/O during interactive sessions

Trade-off: If the dataset changes, the server must be restarted. For a demo system, this is acceptable.

---

## Database Choice

### Primary: In-Memory NetworkX Graph (Runtime)

The core database for this system is an **in-memory directed graph** built from the JSONL source files at startup.

**Why not a traditional database like PostgreSQL or SQLite?**

The assignment asks for a *graph-based* system where relationships are first-class citizens. Relational databases model relationships as foreign keys — you have to JOIN tables to traverse them. For a query like "trace the full flow of billing document 91150187", a relational approach requires:

```sql
SELECT o.*, d.*, b.*, j.*
FROM orders o
JOIN deliveries d ON o.order_id = d.order_id
JOIN billing b ON d.delivery_id = b.delivery_id
JOIN journal j ON b.billing_id = j.billing_id
WHERE b.billing_id = '91150187'
```

The same query on a graph is a single path traversal — follow edges from the billing node backward/forward. This is O(depth) vs O(n log n) for indexed JOINs.

**Why not a dedicated graph database like Neo4j?**

Neo4j would be the production choice. For this assignment, NetworkX was chosen because:

- Zero infrastructure setup (no Docker, no external service)
- The dataset fits comfortably in memory
- NetworkX's API is expressive enough for all required queries
- The architecture is designed so NetworkX can be swapped for Neo4j's Python driver with minimal changes to the query layer

**Why not SQLite?**

SQLite would work for simple lookups but adds no value over a dictionary for this use case. The O2C flow is graph-shaped, not table-shaped. Forcing it into SQLite would require multiple JOINs for every trace query and provide no path-finding capability.

### Source Files: JSONL

Raw data is stored as JSONL (JSON Lines) files in the `data/sap-o2c-data/` directory. JSONL was kept as-is rather than migrating to a database because:

- It is the provided format
- Pandas reads it in a single call
- The files are only read once at startup

---

## Graph Modeling

### Node Types

| Node Prefix | Entity | Example ID |
|---|---|---|
| `order_` | Sales Order Header | `order_1000012` |
| `delivery_` | Outbound Delivery Header | `delivery_80012345` |
| `invoice_` | Billing Document Header | `invoice_91150187` |
| `payment_` | Payment Document | `payment_1400000001` |
| `customer_` | Customer Master | `customer_C1001` |
| `product_` | Material / Product | `product_MAT-001` |

Prefixing node IDs with their entity type (e.g. `order_1000012` instead of just `1000012`) is a deliberate design decision. Without prefixes, numeric IDs from different entity types would collide in the graph (an order and a delivery could share the same numeric ID).

### Edge Types

| From | To | Relationship |
|---|---|---|
| Sales Order | Delivery | `HAS_DELIVERY` |
| Delivery | Invoice | `HAS_INVOICE` |
| Invoice | Payment | `HAS_PAYMENT` |
| Customer | Sales Order | `PLACED_ORDER` |
| Sales Order | Product | `CONTAINS_PRODUCT` |

All edges are directed, modeling the actual business process flow direction.

### Broken Flow Detection

A broken flow is defined as a node that has incoming edges but no outgoing edges where one is expected. For example:

- An order node with a delivery edge but no invoice edge = "delivered but not billed"
- A delivery node with no invoice edge = "incomplete O2C chain"

These are detected via `G.out_degree(node) == 0` checks filtered by node type.

---

## LLM Integration & Prompting Strategy

### LLM Provider: Groq (llama-3.1-8b-instant)

Groq was chosen for its **free tier** and **extremely low latency** (<500ms for classification prompts). The `llama-3.1-8b-instant` model is fast and accurate enough for single-word intent classification.

### Prompting Strategy: Constrained Classification

The LLM is not asked to generate SQL or perform free-form reasoning. It is given a strict, constrained prompt that asks for exactly one word from a fixed vocabulary:

```python
prompt = f"""
Classify the query into ONE word:
orders OR deliveries OR trace

Query: {user_query}

Answer only one word.
"""
```

**Why this approach over open-ended generation?**

Open-ended LLM responses are hard to parse reliably. A response like "I think you want to see deliveries" would require regex, NLP parsing, or a second LLM call to extract the intent. By constraining to a single word from a known vocabulary, the intent can be used directly in an `if/elif` chain.

**Why not few-shot examples?**

The classification task is simple enough that zero-shot works reliably. Adding few-shot examples would increase token usage and latency on the Groq free tier with no meaningful accuracy improvement for a 3-class problem.

**Fallback Chain**

If the LLM returns `"unknown"` or an unexpected value, the system falls back to keyword matching on the original query:

```python
if intent == "unknown":
    if "order" in q_lower:      intent = "orders"
    elif "deliver" in q_lower:  intent = "deliveries"
    elif "trace" in q_lower:    intent = "trace"
```

This two-layer approach (LLM first, keyword fallback second) ensures the system degrades gracefully if the Groq API is slow, rate-limited, or returns a malformed response.

### Intent → Query Mapping

| Intent | Graph Operation | Description |
|---|---|---|
| `orders` | Filter nodes by `"order" in node_id` | Returns first 10 order nodes |
| `deliveries` | Filter nodes by `"delivery" in node_id` | Returns first 10 delivery nodes |
| `trace` | Iterate `G.edges()` | Returns first 10 directed edges as from/to pairs |

The frontend renders edges as directed arrows for `trace` intent and as isolated dots for `orders`/`deliveries`.

---

## Guardrails

The system implements multiple layers of guardrails to prevent misuse and off-topic usage.

### 1. LLM-Level Intent Constraint

The LLM prompt explicitly restricts the classification vocabulary to domain-relevant intents (`orders`, `deliveries`, `trace`). Any query outside the O2C domain will return `"unknown"` — the model has no valid classification to apply to questions like "write me a poem" or "what is the capital of France."

### 2. Intent Validation Before Query Execution

After the LLM returns an intent, the backend validates it before any graph operation:

```python
if "order" in intent:
    # execute orders query
elif "deliver" in intent:
    # execute deliveries query
elif "trace" in intent or "flow" in intent:
    # execute trace query
else:
    return {"message": "Could not understand query"}
```

If none of the valid intents match, the system returns a polite refusal rather than attempting to execute unknown logic.

### 3. Query Scoping

All graph queries are scoped to the loaded dataset. There is no mechanism for the LLM to access external data sources, the internet, or system resources. The graph `G` is a closed, bounded data structure populated only from the JSONL source files.

### 4. Error Isolation

All query logic is wrapped in a try/except block. If graph traversal throws (e.g. the graph is empty, a node ID is malformed, the JSONL files are missing), the error is caught and returned as a structured JSON error response rather than crashing the server or leaking stack traces to the frontend:

```python
except Exception as e:
    return {"error": str(e)}
```

### 5. CORS Configuration

The API is configured with `allow_origins=["*"]` for development. In production, this should be restricted to the specific frontend domain to prevent cross-origin abuse.

### 6. Result Limiting

All query results are capped (`.edges[:10]`, `[:10]`) to prevent the frontend from being overwhelmed with thousands of nodes that would make the graph unreadable and the browser unresponsive.

### 7. Frontend-Level Rejection Display

When the backend returns `"Could not understand query"`, the frontend displays it in the chat panel with an error style. The user is not given an empty state or a confusing blank graph — they receive explicit feedback that their query was out of scope.

---

## Project Structure

```
project/
├── main.py              # FastAPI backend — graph loading, LLM integration, query API
├── index.html           # Frontend — vis-network graph + chat interface
├── data/
│   └── sap-o2c-data/
│       ├── sales_order_headers/      # JSONL files
│       ├── outbound_delivery_headers/ # JSONL files
│       └── billing_document_headers/  # JSONL files
└── README.md
```

---

## Setup & Running

### Prerequisites

- Python 3.9+
- A Groq API key (free at https://console.groq.com)

### Install Dependencies

```bash
pip install fastapi uvicorn pandas networkx groq
```

### Configure API Key

Open `main.py` and replace:

```python
client = Groq(api_key="YOUR_GROQ_API_KEY_HERE")
```

### Run the Backend

```bash
uvicorn main:app --reload --port 8000
```

### Open the Frontend

Open `index.html` directly in your browser. No build step required.

---

## Example Queries

| Query | Intent | What Happens |
|---|---|---|
| "Show me all orders" | `orders` | Returns 10 order nodes, renders as dots on graph |
| "List deliveries" | `deliveries` | Returns 10 delivery nodes |
| "Trace the full flow" | `trace` | Returns 10 directed edges, renders with arrows |
| "Find the billing document flow" | `trace` | LLM maps to trace intent |
| "Which orders are there?" | `orders` | LLM maps to orders intent |
| "Write me a poem" | `unknown` | Returns "Could not understand query" |
| "What is 2+2?" | `unknown` | Returns "Could not understand query" |

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Backend Framework | FastAPI | Async, fast, Pydantic validation |
| Graph Engine | NetworkX DiGraph | In-memory, path-finding, no infra overhead |
| LLM Provider | Groq (llama-3.1-8b-instant) | Free tier, <500ms latency |
| Data Format | JSONL | Provided format, Pandas native |
| Graph Visualization | vis-network | Physics simulation, click/hover events |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, single file |
| Fonts | DM Sans + DM Mono | Clean, professional, readable |

---

## Design Tradeoffs Summary

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Graph store | NetworkX in-memory | Neo4j | Zero infra, fits dataset, swappable |
| LLM task | Intent classification (1 word) | SQL generation | Reliable parsing, no regex needed |
| Frontend | Single HTML file | React SPA | No build step, instant demo |
| Data loading | Lazy singleton | Per-request load | Speed, avoids redundant disk I/O |
| LLM fallback | Keyword matching | Second LLM call | Latency, free tier rate limits |
