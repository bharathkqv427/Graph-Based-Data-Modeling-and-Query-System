from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import networkx as nx
import os
import json
from groq import Groq
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI() 

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

load_dotenv()

# ✅ CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

G = nx.DiGraph()
DATA_LOADED = False

# 🔑 YOUR GROQ KEY
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.get("/")
def home():
    return {"message": "Graph API Running 🚀"}


# ---------------- LOAD JSONL ----------------
def load_jsonl_folder(folder_path):
    data = []

    if not os.path.exists(folder_path):
        print("❌ Path not found:", folder_path)
        return pd.DataFrame()

    for file in os.listdir(folder_path):
        if file.endswith(".jsonl"):
            full_path = os.path.join(folder_path, file)

            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except:
                        pass

    return pd.DataFrame(data)


# ---------------- BUILD GRAPH ----------------
def build_graph():
    global DATA_LOADED

    if DATA_LOADED:
        return

    print("Loading graph...")

    orders = load_jsonl_folder("data/sap-o2c-data/sales_order_headers")
    deliveries = load_jsonl_folder("data/sap-o2c-data/outbound_delivery_headers")
    invoices = load_jsonl_folder("data/sap-o2c-data/billing_document_headers")

    print("Orders:", len(orders))
    print("Deliveries:", len(deliveries))
    print("Invoices:", len(invoices))

    # ✅ CLEAN DATA (NO NAN)
    order_ids = orders.iloc[:, 0].dropna().astype(str).tolist()
    delivery_ids = deliveries.iloc[:, 0].dropna().astype(str).tolist()
    invoice_ids = invoices.iloc[:, 0].dropna().astype(str).tolist()

    # Add nodes
    for oid in order_ids:
        G.add_node(f"order_{oid}", type="order")

    for did in delivery_ids:
        G.add_node(f"delivery_{did}", type="delivery")

    for iid in invoice_ids:
        G.add_node(f"invoice_{iid}", type="invoice")

    # Add edges
    min_len = min(len(order_ids), len(delivery_ids), len(invoice_ids))

    for i in range(min_len):
        G.add_edge(f"order_{order_ids[i]}", f"delivery_{delivery_ids[i]}")
        G.add_edge(f"delivery_{delivery_ids[i]}", f"invoice_{invoice_ids[i]}")

    DATA_LOADED = True
    print("Graph Loaded Successfully ✅")


# ---------------- LLM ----------------
def ask_llm(user_query):
    try:
        prompt = f"""
Classify the query into ONE word:
orders OR deliveries OR trace

Query: {user_query}

Answer only one word.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content.strip().lower()
        print("LLM Response:", result)

        return result

    except Exception as e:
        print("LLM error:", e)
        return "unknown"


# ---------------- QUERY ----------------
@app.get("/query")
def query_graph(q: str):
    try:
        build_graph()

        intent = ask_llm(q)
        print("Intent:", intent)

        # Fallback
        if intent == "unknown":
            q_lower = q.lower()
            if "order" in q_lower:
                intent = "orders"
            elif "deliver" in q_lower:
                intent = "deliveries"
            elif "trace" in q_lower or "flow" in q_lower:
                intent = "trace"

        # Orders
        if "order" in intent:
            result = [n for n in G.nodes if "order" in n][:10]
            return {"intent": "orders", "data": result}

        # Deliveries
        elif "deliver" in intent:
            result = [n for n in G.nodes if "delivery" in n][:10]
            return {"intent": "deliveries", "data": result}

        # Trace
        elif "trace" in intent or "flow" in intent:
            result = []
            for edge in list(G.edges)[:10]:
                result.append({"from": edge[0], "to": edge[1]})

            return {"intent": "trace", "data": result}

        return {"message": "Could not understand query"}

    except Exception as e:
        return {"error": str(e)}
