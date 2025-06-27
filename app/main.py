import re
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import TypedDict
from langgraph.graph import StateGraph
import pandas as pd
import fitz  # PyMuPDF
import io
import pytesseract
from PIL import Image
from docx import Document
from langdetect import detect
from googletrans import Translator

# ─── Setup ──────────────────────────────────────────────────────────────────────

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")
translator = Translator()

CUSTOMER_DB = pd.read_csv("data/customers.csv")
ACTIONS_DB = pd.read_csv("data/actions.csv")
VALID_ACTIONS = {a.lower().replace(' ', '_'): a for a in ACTIONS_DB['action_name']}

# ─── LangGraph State ────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    file: UploadFile
    text: str
    national_id: str
    action: str
    customer_id: str
    result: str
    error: str

# ─── Core Utility Functions ─────────────────────────────────────────────────────

def extract_text_from_any_file(file: UploadFile) -> str:
    try:
        filename = file.filename.lower()
        file.file.seek(0)
        file_bytes = file.file.read()

        if filename.endswith(".pdf"):
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                return "".join([page.get_text() for page in doc])

        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs])

        elif filename.endswith(".txt"):
            return file_bytes.decode("utf-8")

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(image)

        else:
            return "Unsupported file type"

    except Exception as e:
        print("Error in extract_text_from_any_file:", e)
        raise

def normalize_language(text: str) -> str:
    try:
        detected = detect(text)
        print("Detected language:", detected)
        if detected != "en":
            translated = translator.translate(text, src=detected, dest="en").text
            print("Translated text:", translated)
            return translated
        return text
    except Exception as e:
        print("Language detection/translation error:", e)
        return text

def extract_entities_with_regex(text: str) -> dict:
    lowered_text = text.lower()

    national_id_match = re.search(
        r"(?:national\s*id|id\s*no\.?|identification\s*number|nid|id number)[^\d]{0,10}(\d{8,20})",
        lowered_text,
        re.IGNORECASE
    )

    action_keywords = {
        "freeze_funds": ["freeze", "block", "hold", "suspend", "restrict"],
        "release_funds": ["release", "unfreeze", "lift", "reactivate", "terminate"]
    }

    action = None
    for act_key, synonyms in action_keywords.items():
        for word in synonyms:
            if re.search(rf"\b{word}\b", lowered_text):
                action = act_key
                break
        if action:
            break

    return {
        "national_id": national_id_match.group(1) if national_id_match else None,
        "action": action
    }

# ─── Action Functions ───────────────────────────────────────────────────────────

def freeze_funds(customer_id): return f"Funds frozen for customer {customer_id}"
def release_funds(customer_id): return f"Funds released for customer {customer_id}"

action_function_map = {
    "freeze_funds": freeze_funds,
    "release_funds": release_funds,
}

# ─── LangGraph Nodes ────────────────────────────────────────────────────────────

def extract_text_node(state: GraphState) -> GraphState:
    try:
        file = state['file']
        raw_text = extract_text_from_any_file(file)
        translated_text = normalize_language(raw_text)
        state['text'] = translated_text
        print("Extracted and normalized text:", translated_text)
    except Exception as e:
        state['error'] = f"Failed to extract text: {str(e)}"
        print("Error in extract_text_node:", e)
    return state

def parse_text_node(state: GraphState) -> GraphState:
    try:
        text = state.get("text", "")
        parsed = extract_entities_with_regex(text)
        state['national_id'] = parsed.get("national_id")
        state['action'] = parsed.get("action")
        print("Parsed entities:", parsed)
    except Exception as e:
        state['error'] = f"Failed to parse text: {str(e)}"
        print("Error in parse_text_node:", e)
    return state

def validate_customer_node(state: GraphState) -> GraphState:
    try:
        national_id = state.get("national_id")
        if not national_id:
            state['error'] = "No national ID detected in the document."
            return state

        national_id = national_id.strip()
        CUSTOMER_DB['national_id'] = CUSTOMER_DB['national_id'].astype(str).str.strip()
        match = CUSTOMER_DB[CUSTOMER_DB['national_id'] == national_id]

        if not match.empty:
            state['customer_id'] = match.iloc[0]['customer_id']
            print(f"Customer matched: {state['customer_id']}")
        else:
            state['error'] = f"National ID {national_id} not found in bank records. Order discarded."
            print(state['error'])
    except Exception as e:
        state['error'] = f"Customer validation failed: {str(e)}"
        print("Error in validate_customer_node:", e)
    return state

def execute_action_node(state: GraphState) -> GraphState:
    try:
        if 'error' in state:
            return state
        action = state.get("action")
        customer_id = state.get("customer_id")
        if action not in action_function_map:
            state['error'] = f"Action '{action}' is not supported."
        else:
            result = action_function_map[action](customer_id)
            state['result'] = result
            print("Executed action result:", result)
    except Exception as e:
        state['error'] = f"Action execution failed: {str(e)}"
        print("Error in execute_action_node:", e)
    return state

# ─── LangGraph Workflow ─────────────────────────────────────────────────────────

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

# ─── Web Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_doc")
async def process_doc(request: Request, file: UploadFile = File(...)):
    try:
        print("Received file:", file.filename)
        initial_state: GraphState = {"file": file}
        print("Invoking LangGraph...")
        result: GraphState = graph.invoke(initial_state)
        print("LangGraph result:", result)

        if 'error' in result:
            print("Processing error:", result['error'])
            return templates.TemplateResponse("index.html", {
                "request": request,
                "result": result['error']
            })

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result.get("result"),
            "customer_id": result.get("customer_id"),
            "action": result.get("action")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Exception in /process_doc:", e)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": "Internal server error. Please check server logs."
        })
