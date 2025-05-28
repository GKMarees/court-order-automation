from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import fitz  # PyMuPDF
import re
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

CUSTOMER_DB = pd.read_csv("data/customers.csv")
ACTIONS_DB = pd.read_csv("data/actions.csv")
VALID_ACTIONS = {a.lower().replace(' ', '_'): a for a in ACTIONS_DB['action_name']}

# Define state schema for LangGraph
class GraphState(TypedDict, total=False):
    file: UploadFile
    text: str
    national_id: str
    action: str
    customer_id: str
    result: str
    error: str

def freeze_funds(customer_id):
    return f"Funds frozen for customer {customer_id}"

def release_funds(customer_id):
    return f"Funds released for customer {customer_id}"

action_function_map = {
    "freeze_funds": freeze_funds,
    "release_funds": release_funds,
}

def extract_text_node(state: GraphState) -> GraphState:
    pdf_file = state['file']
    text = ""
    with fitz.open(stream=pdf_file.file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    state['text'] = text
    print("EXTRACTED TEXT:", text)
    return state

def parse_text_node(state: GraphState) -> GraphState:
    text = state.get("text", "")
    national_id_match = re.search(r"(?:National ID|ID No\.|identification number)[^\d]*(\d{10})", text, re.IGNORECASE)
    action_match = re.search(r"(?i)(freeze|release|suspend|transfer)[^\.\n]*", text)
    national_id = national_id_match.group(1) if national_id_match else None
    raw_action = action_match.group(1).lower() if action_match else None
    action = None
    if raw_action:
        if "freeze" in raw_action:
            action = "freeze_funds"
        elif "release" in raw_action:
            action = "release_funds"
    state['national_id'] = national_id
    state['action'] = action
    print("PARSED ID:", national_id, "ACTION:", action) 
    return state

def validate_customer_node(state: GraphState) -> GraphState:
    national_id = state.get("national_id", "").strip()
    national_id = national_id.replace(" ", "")
    CUSTOMER_DB['national_id'] = CUSTOMER_DB['national_id'].astype(str).str.strip()
    
    match = CUSTOMER_DB[CUSTOMER_DB['national_id'] == national_id]
    if not match.empty:
        state['customer_id'] = match.iloc[0]['customer_id']
    else:
        state['error'] = f"National ID {national_id} not found in bank records. Order discarded."
    return state

def execute_action_node(state: GraphState) -> GraphState:
    if 'error' in state:
        return state
    action = state.get("action")
    customer_id = state.get("customer_id")
    if action not in VALID_ACTIONS:
        state['error'] = f"Action '{action}' is not recognized. Stopping processing."
    else:
        result = action_function_map[action](customer_id)
        state['result'] = result
    return state

# Build LangGraph with proper schema
graph_builder = StateGraph(GraphState)
graph_builder.add_node("extract_text", extract_text_node)
graph_builder.add_node("parse_text", parse_text_node)
graph_builder.add_node("validate_customer", validate_customer_node)
graph_builder.add_node("execute_action", execute_action_node)

graph_builder.set_entry_point("extract_text")
graph_builder.add_edge("extract_text", "parse_text")
graph_builder.add_edge("parse_text", "validate_customer")
graph_builder.add_edge("validate_customer", "execute_action")
graph_builder.set_finish_point("execute_action")

graph = graph_builder.compile()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_doc")
async def process_doc(request: Request, file: UploadFile = File(...)):
    initial_state: GraphState = {"file": file}
    result: GraphState = graph.invoke(initial_state)
    if 'error' in result:
        return templates.TemplateResponse("index.html", {"request": request, "result": result['error']})
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result.get("result"),
        "customer_id": result.get("customer_id"),
        "action": result.get("action")
    })