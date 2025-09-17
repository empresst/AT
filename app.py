

import os
import json
import re
from typing import List, Optional, Tuple, Dict, Any
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import uuid
from datetime import datetime, timedelta
import pytz
import numpy as np
import logging
import spacy
from nltk.corpus import wordnet
import nltk
from cachetools import TTLCache
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import threading
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

# ---------------------
# Setup logging
# ---------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-twin-app")

# ---------------------
# NLTK & spaCy
# ---------------------
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError as e:
    nlp = None
    logger.warning(f"spaCy not installed: {e}. Skipping NLP features.")
try:
    import pytz
except ImportError as e:
    logger.warning(f"pytz not installed: {e}. Skipping timezone features.")
    pytz = None

# ---------------------
# Env & constants
# ---------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
PUBLIC_UI_API_KEY = os.getenv("PUBLIC_UI_API_KEY", "your-secure-api-key")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI missing")

# Session settings
SESSION_TTL_MIN = int(os.getenv("SESSION_TTL_MIN", "4320"))  # 3 days default

# ---------------------
# Globals (thread-safe)
# ---------------------
client: Optional[AsyncIOMotorClient] = None
openai_client: Optional[AsyncOpenAI] = None
faiss_store: Optional[FAISS] = None

db = None
users_col = None
conversations_col = None
journals_col = None
embeddings_col = None
personalities_col = None
errors_col = None
saved_greetings_col = None
greetings_cache_col = None
relationships_col = None
sessions_col = None

mongo_lock = threading.Lock()
openai_lock = threading.Lock()
faiss_lock = threading.Lock()

embedding_cache = TTLCache(maxsize=1000, ttl=3600)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
FAISS_DIR = "faiss_store_v1"
watcher_task: Optional[asyncio.Task] = None

def as_utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=pytz.UTC)

# ---------------------
# Lazy init functions
# ---------------------
async def get_mongo_client() -> AsyncIOMotorClient:
    global client, db, users_col, conversations_col, journals_col, embeddings_col, personalities_col
    global errors_col, saved_greetings_col, greetings_cache_col, relationships_col, sessions_col
    with mongo_lock:
        if client is None:
            client = AsyncIOMotorClient(
                MONGODB_URI,
                tls=True,
                tlsAllowInvalidCertificates=True,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                tz_aware=True,
                tzinfo=pytz.UTC
            )
            db = client["LF"]
            users_col = db["users"]
            conversations_col = db["conversations"]
            journals_col = db["journal_entries"]
            embeddings_col = db["embeddings"]
            personalities_col = db["personalities"]
            errors_col = db["errors"]
            saved_greetings_col = db["saved_greetings"]
            greetings_cache_col = db["greetings"]
            relationships_col = db["relationships"]
            sessions_col = db["sessions"]

    # indexes
    await conversations_col.create_index([("user_id", 1), ("timestamp", -1)])
    await conversations_col.create_index([("speaker_id", 1), ("target_id", 1), ("timestamp", -1)])
    await conversations_col.create_index([("content", "text")])
    await journals_col.create_index([("user_id", 1), ("timestamp", -1)])
    await journals_col.create_index([("content", "text")])
    await embeddings_col.create_index([("item_id", 1), ("item_type", 1)])
    await personalities_col.create_index([("user_id", 1)])
    await errors_col.create_index([("timestamp", -1)])
    await saved_greetings_col.create_index([("target_id", 1), ("bot_role", 1), ("timestamp", -1)])
    await greetings_cache_col.create_index([("key", 1), ("timestamp", -1)])
    await relationships_col.create_index([("user_id", 1), ("other_user_id", 1)], unique=True)
    await sessions_col.create_index("expires_at", expireAfterSeconds=0)

    return client

async def get_openai_client() -> AsyncOpenAI:
    global openai_client
    with openai_lock:
        if openai_client is None:
            openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return openai_client

async def ensure_faiss_store():
    global faiss_store
    with faiss_lock:
        if faiss_store is None:
            if os.path.isdir(FAISS_DIR):
                try:
                    faiss_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                    return
                except Exception as e:
                    logger.warning(f"FAISS load failed: {e}. Rebuilding...")
    await initialize_faiss_store()

async def initialize_faiss_store():
    global faiss_store
    await get_mongo_client()
    with faiss_lock:
        if os.path.isdir(FAISS_DIR):
            try:
                faiss_store = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                return
            except Exception:
                pass

    emb_data = await embeddings_col.find().to_list(length=None)
    docs: List[Document] = []
    for emb in emb_data:
        try:
            item_id = emb.get("item_id")
            item_type = emb.get("item_type")
            if not item_id or not item_type:
                await embeddings_col.delete_one({"_id": emb["_id"]})
                continue
            col = conversations_col if item_type == "conversation" else journals_col
            id_field = "conversation_id" if item_type == "conversation" else "entry_id"
            base = await col.find_one({id_field: item_id})
            if not base:
                await embeddings_col.delete_one({"item_id": item_id, "item_type": item_type})
                continue

            content = emb.get("content", base.get("content", ""))
            if not content:
                await embeddings_col.delete_one({"item_id": item_id, "item_type": item_type})
                continue

            speaker_name = emb.get("speaker_name")
            target_name = emb.get("target_name")
            metadata = {
                "item_id": item_id,
                "item_type": item_type,
                "user_id": emb.get("user_id", []),
                "speaker_id": emb.get("speaker_id"),
                "target_id": emb.get("target_id"),
                "speaker_name": speaker_name,
                "target_name": target_name,
                "timestamp": as_utc_aware(emb.get("timestamp"))
            }
            docs.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            await embeddings_col.delete_one({"_id": emb["_id"]})

    with faiss_lock:
        if docs:
            faiss_store = FAISS.from_documents(docs, embeddings)
        else:
            faiss_store = FAISS.from_texts(["empty"], embeddings)
        faiss_store.save_local(FAISS_DIR)

# ---------------------
# Security: session tokens
# ---------------------
async def create_session(user_id: str) -> str:
    await get_mongo_client()
    token = str(uuid.uuid4())
    now = datetime.now(pytz.UTC)
    await sessions_col.insert_one({
        "token": token,
        "user_id": user_id,
        "created_at": now,
        "expires_at": now + timedelta(minutes=SESSION_TTL_MIN)
    })
    return token

async def require_session(x_session_token: str = Header(...)) -> Dict[str, Any]:
    await get_mongo_client()
    sess = await sessions_col.find_one({"token": x_session_token})
    if not sess:
        raise HTTPException(status_code=401, detail="Invalid session")
    if as_utc_aware(sess["expires_at"]) < datetime.now(pytz.UTC):
        raise HTTPException(status_code=401, detail="Session expired")
    user = await users_col.find_one({"user_id": sess["user_id"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"token": x_session_token, "user": user}

# ---------------------
# Password hashing (no external deps)
# ---------------------
import hashlib, secrets, base64

def hash_password(password: str) -> Dict[str, str]:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return {"salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(dk).decode()}

def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    salt = base64.b64decode(salt_b64.encode())
    expected = base64.b64decode(hash_b64.encode())
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return secrets.compare_digest(dk, expected)

# ---------------------
# Connection manager (WebSockets)
# ---------------------
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}  # user_id -> socket
        self.lock = asyncio.Lock()

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active[user_id] = websocket

    async def disconnect(self, user_id: str):
        async with self.lock:
            self.active.pop(user_id, None)

    async def is_online(self, user_id: str) -> bool:
        async with self.lock:
            return user_id in self.active

    async def send_to(self, user_id: str, data: dict):
        async with self.lock:
            ws = self.active.get(user_id)
        if ws:
            await ws.send_json(data)

    async def broadcast_presence(self):
        # broadcast the list of online users
        async with self.lock:
            online = list(self.active.keys())
            sockets = list(self.active.values())
        payload = {"type": "presence", "online": online}
        for ws in sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                pass

manager = ConnectionManager()

# ---------------------
# FastAPI app
# ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global watcher_task
    await initialize_db()
    watcher_task = asyncio.create_task(watch_collections())
    yield
    if watcher_task:
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
    if client:
        client.close()

app = FastAPI(title="Chatbot AI Twin API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Pydantic models
# ---------------------
class MessageRequest(BaseModel):
    speaker_id: str
    target_id: str
    bot_role: str
    user_input: str

class MessageResponse(BaseModel):
    response: str
    error: Optional[str] = None

class SignupRequest(BaseModel):
    username: str
    display_name: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RelationshipSetRequest(BaseModel):
    other_user_id: str
    relation: str  # daughter, son, mother, father, sister, brother, wife, husband, friend

# ---------------------
# HTML UI (login + users + relationships + multi chat + AI toggle)
# ---------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>AI Twin Chat</title>
<style>
* { box-sizing: border-box }
:root { --bg:#0f172a; --fg:#fff; --muted:#64748b; --border:#e2e8f0; --card:#f8fafc; }
body { margin:0; font-family: Inter, system-ui, Arial }
header { padding: 10px 16px; background:var(--bg); color:var(--fg); display:flex; justify-content:space-between; align-items:center }
main { display:flex; height: calc(100vh - 56px) }
#sidebar { width:340px; border-right:1px solid var(--border); padding:12px; overflow:auto }
#content { flex:1; display:flex; flex-direction:column }
section { margin-bottom:16px }
h3 { margin:8px 0 }
input, select, button, textarea { padding:10px; margin:6px 0; width:100%; font-size:14px }
textarea { resize:none; min-height:48px; max-height:160px; line-height:20px; }
button { cursor:pointer }
.user-item { padding:8px; border:1px solid var(--border); border-radius:8px; margin:6px 0; display:flex; gap:8px; align-items:center; justify-content:space-between; background:#fff }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#e2e8f0 }
.badge.online { background:#bbf7d0 }
.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#fde68a; color:#92400e; margin-left:6px }
.chatgrid { flex:1; display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:12px; padding:12px; overflow:auto }
.chatbox { border:1px solid var(--border); border-radius:10px; display:flex; flex-direction:column; min-height:360px; background:#fff }
.chatbox header { background:var(--card); color:#0f172a; font-weight:600; padding:8px 10px; display:flex; align-items:center; justify-content:space-between }
.header-right { display:flex; align-items:center; gap:6px }
.messages { flex:1; padding:10px; overflow:auto; font-size:14px; background:#fafafa }
.msg { margin:8px 0; padding:8px 10px; border-radius:8px; max-width:92%; box-shadow:0 1px 0 rgba(0,0,0,.04) }
.msg.you { background:#dbeafe; align-self:flex-end }
.msg.them { background:#dcfce7 }
.msg.ai { background:#fef3c7 }
.meta { font-size:11px; color:var(--muted); margin-top:4px }
.typing { font-size:12px; color:var(--muted); margin:6px 0 0 2px }
.actions { padding:8px; display:flex; gap:8px; border-top:1px solid var(--border); background:#fff }
.actions textarea { flex:1; border:1px solid var(--border); border-radius:8px; padding:10px 12px; }
.actions button { width:auto; padding:10px 14px; border-radius:8px; background:#0ea5e9; color:#fff; border:none }
.actions button:disabled { opacity:.6; cursor:not-allowed }
.small { font-size:12px; color:var(--muted) }
.form-card { border:1px solid var(--border); border-radius:10px; padding:12px; background:#fff }
.row { display:flex; gap:8px }
.row > * { flex:1 }
hr { border:0; border-top:1px solid var(--border); margin:12px 0 }
</style>
</head>
<body>
<header>
  <div>AI Twin Chat</div>
  <div id="whoami" class="small"></div>
</header>
<main>
  <div id="sidebar">
    <section id="auth">
      <div class="form-card">
        <h3>Login</h3>
        <input id="loginUsername" placeholder="username"/>
        <input id="loginPassword" placeholder="password" type="password"/>
        <button id="loginBtn">Login</button>
      </div>
      <hr/>
      <div class="form-card">
        <h3>Sign up</h3>
        <input id="signupUsername" placeholder="username"/>
        <input id="signupDisplayName" placeholder="display name"/>
        <input id="signupPassword" placeholder="password" type="password"/>
        <button id="signupBtn">Create account</button>
      </div>
      <div style="margin-top:10px">
        <div class="small">Server key for UI (x-api-key)</div>
        <input id="gatewayKey" placeholder="x-api-key (server key)" />
      </div>
    </section>

    <section id="me" style="display:none">
      <div class="form-card">
        <h3>Me</h3>
        <div id="meInfo"></div>
        <div class="row">
          <label class="small" style="display:flex; align-items:center; gap:8px">
            <input type="checkbox" id="aiToggle"/>
            AI respond for me
          </label>
          <button id="logoutBtn" style="background:#ef4444">Logout</button>
        </div>
      </div>
    </section>

    <section id="users" style="display:none">
      <h3>Users</h3>
      <div id="usersList"></div>
    </section>

    <section id="rel" style="display:none">
      <h3>Set Relationship</h3>
      <div class="row">
        <select id="relOther"></select>
        <select id="relKind">
          <option>daughter</option><option>son</option><option>mother</option><option>father</option>
          <option>sister</option><option>brother</option><option>wife</option><option>husband</option>
          <option>friend</option>
        </select>
      </div>
      <button id="relSave">Save</button>
      <div id="relStatus" class="small"></div>
    </section>
  </div>

  <div id="content">
    <div class="chatgrid" id="chatGrid"></div>
  </div>
</main>

<script>
let API = location.origin;
let API_KEY = "";
let SESSION = "";
let ME = null;
let WS = null;
const chatBoxes = new Map(); // other_user_id -> {box, area, btn, typingEl, aiExpected}

function el(id){ return document.getElementById(id) }

function setAuthVisible(loggedIn){
  el('auth').style.display = loggedIn ? 'none' : 'block';
  el('me').style.display = loggedIn ? 'block' : 'none';
  el('users').style.display = loggedIn ? 'block' : 'none';
  el('rel').style.display = loggedIn ? 'block' : 'none';
}

function autoresizeTA(ta){
  ta.style.height = 'auto';
  ta.style.height = Math.min(160, Math.max(48, ta.scrollHeight)) + 'px';
}

async function req(path, method='GET', body=null){
  const headers = {'Content-Type':'application/json'};
  if(API_KEY) headers['x-api-key']=API_KEY;
  if(SESSION) headers['x-session-token']=SESSION;
  const res = await fetch(API+path, {method, headers, body: body?JSON.stringify(body):undefined});
  if(!res.ok){ throw new Error(await res.text()) }
  return res.json();
}

function renderMe(){
  el('meInfo').innerHTML = `
    <div><b>${ME.display_name}</b> <span class="small">(@${ME.username})</span></div>
    <div class="small">user_id: ${ME.user_id}</div>
    <div class="small">AI: ${ME.ai_enabled ? 'ON' : 'OFF'}</div>
  `;
  el('whoami').innerText = `${ME.display_name} (@${ME.username})`;
  el('aiToggle').checked = !!ME.ai_enabled;
}

async function refreshUsers(){
  const data = await req('/users/list');
  const container = el('usersList');
  const relSel = el('relOther');
  container.innerHTML = '';
  relSel.innerHTML = '';
  (data.users || []).filter(u=>u.user_id!==ME.user_id).forEach(u=>{
     const div = document.createElement('div');
     div.className='user-item';
     const badge = `<span class="badge ${u.online?'online':''}">${u.online?'online':'offline'}</span>`;
     const ai = u.ai_enabled ? `<span class="pill">AI replies</span>` : '';
     div.innerHTML = `
       <div>
         <div><b>${u.display_name}</b> <span class="small">(@${u.username})</span> ${ai}</div>
         <div class="small">rel: ${u.relation || '-'}</div>
       </div>
       <div>
         ${badge}
         <button data-id="${u.user_id}">Chat</button>
       </div>`;
     div.querySelector('button').onclick=()=>openChat(u);
     container.appendChild(div);

     const opt = document.createElement('option');
     opt.value = u.user_id; opt.textContent = `${u.display_name} (@${u.username})`;
     relSel.appendChild(opt);
  });
}

function ensureBox(u){
  if(chatBoxes.has(u.user_id)) return chatBoxes.get(u.user_id);
  const div = document.createElement('div');
  div.className='chatbox';
  div.innerHTML = `
    <header>
      <div>${u.display_name} <span class="small">(@${u.username})</span></div>
      <div class="header-right">
        <span class="badge ${u.online?'online':''}" id="on_${u.user_id}">${u.online?'online':'offline'}</span>
        ${u.ai_enabled?'<span class="pill">AI replies</span>':''}
      </div>
    </header>
    <div class="messages" id="msg_${u.user_id}"></div>
    <div class="typing" id="typing_${u.user_id}" style="display:none">AI is typing…</div>
    <div class="actions">
      <textarea placeholder="Write a message…" id="inp_${u.user_id}"></textarea>
      <button id="send_${u.user_id}">Send</button>
    </div>
  `;
  el('chatGrid').appendChild(div);
  const area = el('inp_'+u.user_id);
  const btn = el('send_'+u.user_id);
  const typingEl = el('typing_'+u.user_id);
  area.addEventListener('input', ()=>autoresizeTA(area));
  area.addEventListener('keypress', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendTo(u.user_id); }});
  btn.onclick=()=>sendTo(u.user_id);
  chatBoxes.set(u.user_id, {box:div, area, btn, typingEl, aiExpected: !!u.ai_enabled});
  return chatBoxes.get(u.user_id);
}

async function openChat(u){
  const {area} = ensureBox(u);
  // load last 30 messages
  const res = await req(`/conversations/with/${u.user_id}?limit=30`);
  const msgs = res.messages || [];
  const pane = el('msg_'+u.user_id);
  pane.innerHTML='';
  msgs.forEach(m=>appendMsg(u.user_id, m));
  area.focus();
}

function appendMsg(other_id, m, localEcho=false){
  const pane = el('msg_'+other_id);
  if(!pane) return;
  const wrapper = document.createElement('div');
  const who = localEcho ? 'you' : (m.speaker_id===ME.user_id ? 'you' : (m.source==='ai_twin' ? 'ai' : 'them'));
  wrapper.className = `msg ${who}`;
  const when = new Date(m.timestamp).toLocaleString();
  wrapper.innerHTML = `${m.content}<div class="meta">${when}${localEcho?' • ✓ Sent':''}</div>`;
  pane.appendChild(wrapper);
  pane.scrollTop = pane.scrollHeight;
}

function showTyping(other_id, on){
  const elT = el('typing_'+other_id);
  if(!elT) return;
  elT.style.display = on ? 'block' : 'none';
}

async function sendTo(other_id){
  const ref = chatBoxes.get(other_id);
  if(!ref) return;
  const {area, btn, aiExpected} = ref;
  const text = (area.value || '').trim();
  if(!text) return;
  area.value=''; autoresizeTA(area);
  btn.disabled = true;

  // Local echo immediately
  appendMsg(other_id, {content:text, timestamp: new Date().toISOString(), speaker_id: ME.user_id, source:'human'}, true);

  // If we expect AI, show typing until reply arrives
  if(aiExpected) showTyping(other_id, true);

  // Prefer WS if connected, else HTTP
  if(WS && WS.readyState===1){
    WS.send(JSON.stringify({type:'chat', to: other_id, text}));
    // AI reply will arrive as WS "ai" event if enabled
    btn.disabled = false;
  }else{
    try{
      const res = await req('/send_message','POST', {speaker_id: ME.user_id, target_id: other_id, bot_role: 'friend', user_input: text});
      // If server returned an AI reply (HTTP path), append it
      if(res && res.response && res.response !== 'Sent.'){
        appendMsg(other_id, {content: res.response, timestamp: new Date().toISOString(), speaker_id: other_id, source:'ai_twin'});
      }
    }catch(e){
      appendMsg(other_id, {content: '⚠️ Failed to send: '+(e.message||e), timestamp: new Date().toISOString(), speaker_id: other_id, source:'system'});
    }finally{
      showTyping(other_id, false);
      btn.disabled = false;
    }
  }
}

function connectWS(){
  if(WS) try{ WS.close() }catch(e){}
  const qp = new URLSearchParams({ token: SESSION, user_id: ME.user_id });
  WS = new WebSocket(API.replace(/^http/,'ws')+'/ws?'+qp.toString());
  WS.onmessage = (ev)=>{
    try{
      const msg = JSON.parse(ev.data);
      if(msg.type==='presence'){
        // Update online badges quickly
        (msg.online||[]).forEach(uid=>{
          const b = document.getElementById('on_'+uid);
          if(b){ b.classList.add('online'); b.textContent='online'; }
        });
      }else if(msg.type==='chat'){
        appendMsg(msg.from, msg.payload);
      }else if(msg.type==='ai'){
        appendMsg(msg.from, msg.payload);
        showTyping(msg.from, false);
      }
    }catch(e){}
  };
}

document.addEventListener('DOMContentLoaded', ()=>{
  // Server key persistence
  el('gatewayKey').value = localStorage.getItem('gw') || '';
  API_KEY = el('gatewayKey').value;
  el('gatewayKey').addEventListener('change', ()=>{ API_KEY = el('gatewayKey').value; localStorage.setItem('gw', API_KEY) });

  // Login
  el('loginBtn').onclick = async ()=>{
    API_KEY = el('gatewayKey').value;
    const username = el('loginUsername').value.trim();
    const password = el('loginPassword').value.trim();
    const res = await req('/auth/login','POST',{username,password});
    SESSION = res.token; ME = res.user;
    setAuthVisible(true); renderMe(); connectWS(); await refreshUsers();
  };

  // Signup (has its own password field now)
  el('signupBtn').onclick = async ()=>{
    API_KEY = el('gatewayKey').value;
    const username = el('signupUsername').value.trim();
    const display_name = el('signupDisplayName').value.trim() || username;
    const password = el('signupPassword').value.trim();
    if(!username || !password){ alert('Username and password required'); return; }
    await req('/auth/signup','POST',{username,display_name,password});
    alert('Signed up! Now login using the Login form above.');
  };

  // Logout
  el('logoutBtn').onclick = async ()=>{
    await req('/auth/logout','POST',{}); SESSION=''; ME=null; setAuthVisible(false); location.reload();
  };

  // AI toggle
  el('aiToggle').onchange = async (e)=>{
    await req(`/users/me/ai-toggle?enabled=${e.target.checked}`,'PATCH');
    ME.ai_enabled = e.target.checked;
  };

  // Relationship save
  el('relSave').onclick = async ()=>{
    const other = el('relOther').value;
    const relation = el('relKind').value;
    if(!other) return;
    await req('/relationships/set','POST', {other_user_id: other, relation});
    el('relStatus').innerText = 'Saved!';
    setTimeout(()=>{el('relStatus').innerText=''},1500);
    await refreshUsers();
  };

  setAuthVisible(false);
});
</script>
</body>
</html>
    """
    return HTMLResponse(html)

# ---------------------
# Auth routes
# ---------------------
def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != PUBLIC_UI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/auth/signup")
async def signup(req: SignupRequest, _: None = Depends(require_api_key)):
    await get_mongo_client()
    existing = await users_col.find_one({"username": req.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username taken")
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    h = hash_password(req.password)
    now = datetime.now(pytz.UTC)
    doc = {
        "user_id": user_id,
        "username": req.username,
        "display_name": req.display_name,
        "password_salt": h["salt"],
        "password_hash": h["hash"],
        "ai_enabled": False,
        "created_at": now,
        "last_seen": now
    }
    await users_col.insert_one(doc)
    return {"ok": True, "user_id": user_id}

@app.post("/auth/login")
async def login(req: LoginRequest, _: None = Depends(require_api_key)):
    await get_mongo_client()
    user = await users_col.find_one({"username": req.username})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(req.password, user.get("password_salt",""), user.get("password_hash","")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = await create_session(user["user_id"])
    await users_col.update_one({"user_id": user["user_id"]}, {"$set": {"last_seen": datetime.now(pytz.UTC)}})
    return {"token": token, "user": {"user_id": user["user_id"], "username": user["username"], "display_name": user["display_name"], "ai_enabled": user.get("ai_enabled", False)}}

@app.post("/auth/logout")
async def logout(sess=Depends(require_session), _: None = Depends(require_api_key)):
    await sessions_col.delete_one({"token": sess["token"]})
    return {"ok": True}

# ---------------------
# Users / Relationships / Presence
# ---------------------
@app.get("/users/me")
async def users_me(sess=Depends(require_session), _: None = Depends(require_api_key)):
    u = sess["user"]
    return {"user": {"user_id": u["user_id"], "username": u["username"], "display_name": u["display_name"], "ai_enabled": u.get("ai_enabled", False)}}

@app.patch("/users/me/ai-toggle")
async def toggle_ai(enabled: bool = Query(...), sess=Depends(require_session), _: None = Depends(require_api_key)):
    await users_col.update_one({"user_id": sess["user"]["user_id"]}, {"$set": {"ai_enabled": bool(enabled)}})
    return {"ok": True, "ai_enabled": bool(enabled)}

@app.get("/users/list")
async def users_list(sess=Depends(require_session), _: None = Depends(require_api_key)):
    me_id = sess["user"]["user_id"]
    all_users = await users_col.find({}, {"password_hash": 0, "password_salt": 0}).to_list(length=None)
    rels = { r["other_user_id"]: r["relation"] async for r in relationships_col.find({"user_id": me_id}) }
    async with manager.lock:
        online = set(manager.active.keys())
    users = []
    for u in all_users:
        users.append({
            "user_id": u["user_id"],
            "username": u["username"],
            "display_name": u["display_name"],
            "online": u["user_id"] in online,
            "relation": rels.get(u["user_id"]),
            "ai_enabled": bool(u.get("ai_enabled", False))   # <-- added
        })
    return {"users": users}


@app.post("/relationships/set")
async def rel_set(req: RelationshipSetRequest, sess=Depends(require_session), _: None = Depends(require_api_key)):
    # unilateral relationship (A -> B); you can store reciprocal if desired
    me_id = sess["user"]["user_id"]
    await relationships_col.update_one(
        {"user_id": me_id, "other_user_id": req.other_user_id},
        {"$set": {"relation": req.relation, "updated_at": datetime.now(pytz.UTC)}},
        upsert=True
    )
    return {"ok": True}

@app.get("/relationships/with/{other_id}")
async def rel_get(other_id: str, sess=Depends(require_session), _: None = Depends(require_api_key)):
    me_id = sess["user"]["user_id"]
    r = await relationships_col.find_one({"user_id": me_id, "other_user_id": other_id})
    return {"relation": (r or {}).get("relation")}

# ---------------------
# Core AI pieces (same as earlier)
# ---------------------
def preprocess_input(user_input: str) -> str:
    try:
        doc = nlp(user_input)
        key_terms = [t.text.lower() for t in doc if t.pos_ in ["NOUN", "VERB"] and not t.is_stop]
        extra_terms = []
        for term in key_terms:
            syns = wordnet.synsets(term)
            synonyms = set()
            for syn in syns:
                for lemma in syn.lemmas():
                    w = lemma.name().replace('_',' ')
                    if w != term and len(w.split()) <= 2:
                        synonyms.add(w)
            extra_terms.extend(list(synonyms)[:3])
        if extra_terms:
            user_input += " " + " ".join(set(extra_terms[:10]))
        return user_input
    except Exception:
        return user_input

async def get_recent_conversation_history(speaker_id: str, target_id: str, limit: int = 6) -> List[dict]:
    await get_mongo_client()
    pipeline = [
        {"$match": {
            "user_id": {"$all": [speaker_id, target_id]},
            "$or": [{"speaker_id": speaker_id, "target_id": target_id},
                    {"speaker_id": target_id, "target_id": speaker_id}]
        }},
        {"$sort": {"timestamp": -1}},
        {"$limit": limit},
        {"$sort": {"timestamp": 1}}
    ]
    out = []
    async for conv in conversations_col.aggregate(pipeline):
        sp_name = conv.get("speaker_name")
        if not sp_name:
            u = await users_col.find_one({"user_id": conv["speaker_id"]})
            sp_name = (u or {}).get("display_name") or (u or {}).get("username") or conv["speaker_id"]
        raw_ts = as_utc_aware(conv["timestamp"])
        out.append({
            "speaker": sp_name,
            "content": conv["content"],
            "timestamp": raw_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "type": conv.get("type","user_input"),
            "source": conv.get("source", "human"),
            "raw_timestamp": raw_ts,
            "conversation_id": conv["conversation_id"]
        })
    return out

async def generate_personality_traits(user_id: str) -> dict:
    await get_mongo_client()
    convs = [doc async for doc in conversations_col.find({"user_id": {"$all":[user_id]}}).sort("timestamp", -1).limit(500)]
    journals = [doc async for doc in journals_col.find({"user_id": {"$in":[user_id]}}).sort("timestamp", -1).limit(500)]
    data_text = "\n".join([c.get("content","") for c in convs] + [j.get("content","") for j in journals])[:1000]
    if not data_text:
        return {"core_traits": {}, "sub_traits": []}
    cached = await personalities_col.find_one({"user_id": user_id})
    if cached and "traits" in cached:
        return cached["traits"]

    u = await users_col.find_one({"user_id": user_id})
    big_five_prompt = f"""
    Analyze this text from {(u or {}).get('display_name', user_id)}:
    {data_text}
    Return a JSON object with:
    - "core_traits": 5 traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) with scores (0-100) and one-sentence explanations.
    - "sub_traits": 3 unique traits with one-sentence descriptions.
    Ensure the response is concise to fit within 700 tokens.
    """
    traits = None
    for attempt in range(3):
        try:
            resp = await (await get_openai_client()).chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"You are a helpful assistant that generates personality traits."},
                    {"role":"user","content":big_five_prompt}
                ],
                max_tokens=700, temperature=0.7
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r'^```json\\s*|\\s*```$', '', txt, flags=re.MULTILINE).strip()
            traits = json.loads(txt)
            if "core_traits" in traits and "sub_traits" in traits:
                if isinstance(traits["core_traits"], list):
                    traits["core_traits"] = {t["trait"]: {"score": t["score"], "explanation": t["explanation"]} for t in traits["core_traits"]}
                break
        except Exception:
            if attempt == 2:
                traits = {
                    "core_traits": {
                        "Openness":{"score":50,"explanation":"Neutral openness."},
                        "Conscientiousness":{"score":50,"explanation":"Neutral conscientiousness."},
                        "Extraversion":{"score":50,"explanation":"Neutral extraversion."},
                        "Agreeableness":{"score":50,"explanation":"Neutral agreeableness."},
                        "Neuroticism":{"score":50,"explanation":"Neutral neuroticism."}
                    },
                    "sub_traits":[
                        {"trait":"neutral","description":"Shows balanced behavior."},
                        {"trait":"adaptable","description":"Adapts to context."},
                        {"trait":"curious","description":"Engages with data."}
                    ]
                }
    await personalities_col.update_one({"user_id":user_id},{"$set":{"traits":traits}}, upsert=True)
    return traits

async def get_greeting_and_tone(bot_role: str, target_id: str) -> Tuple[str,str]:
    await get_mongo_client()
    key = f"greeting_{target_id}_{bot_role}"
    cached = await greetings_cache_col.find_one({"key": key, "timestamp": {"$gte": datetime.now(pytz.UTC)-timedelta(hours=1)}})
    if cached:
        return cached["greeting"], cached["tone"]

    saved = await saved_greetings_col.find_one({"target_id": target_id, "bot_role": bot_role.lower()}, sort=[("timestamp",-1)])
    if saved:
        return saved["greeting"], "warm, youthful" if bot_role.lower() in ["daughter","son"] else "nurturing, caring"

    defaults = {
        "daughter": ("Hey, Mom", "warm, youthful"),
        "son": ("Hey, Mom", "warm, youthful"),
        "mother": ("Hi, sweetie", "nurturing, caring"),
        "father": ("Hey, kid", "warm, supportive"),
        "sister": ("Yo, sis", "playful, casual"),
        "brother": ("Yo, bro", "playful, casual"),
        "wife": ("Hey, love", "affectionate, conversational"),
        "husband": ("Hey, hon", "affectionate, conversational"),
        "friend": ("Hey, what's good?", "casual, friendly")
    }
    greeting, tone = defaults.get(bot_role.lower(), ("Hey","casual, friendly"))

    traits = await generate_personality_traits(target_id)
    prompt = f"""
    You are generating a greeting for a {bot_role} with traits: {', '.join(traits.get('core_traits', {}).keys())}.
    Return a JSON object: {{"greeting":"short greeting","tone":"tone description"}}
    """
    for attempt in range(3):
        try:
            resp = await (await get_openai_client()).chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"Return only valid JSON with 'greeting' and 'tone' keys."},
                    {"role":"user","content":prompt}
                ], max_tokens=100, temperature=0.5
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r'^```json\\s*|\\s*```$','',txt, flags=re.MULTILINE).strip()
            obj = json.loads(txt)
            if "greeting" in obj and "tone" in obj:
                greeting, tone = obj["greeting"], obj["tone"]
                break
        except Exception:
            if attempt==2: break

    await greetings_cache_col.update_one({"key":key},{"$set":{"greeting":greeting,"tone":tone,"timestamp":datetime.now(pytz.UTC)}}, upsert=True)
    return greeting, tone

async def find_relevant_memories(speaker_id: str, user_id: str, user_input: str, speaker_name: str, max_memories: int = 5) -> List[dict]:
    global faiss_store
    await ensure_faiss_store()
    await get_mongo_client()
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(None, preprocess_input, user_input)
    cache_key = f"input_{hash(processed)}"
    if cache_key in embedding_cache:
        _ = embedding_cache[cache_key]
    else:
        _ = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
        embedding_cache[cache_key] = _

    udoc = await users_col.find_one({"user_id": user_id})
    target_name = (udoc or {}).get("display_name") or (udoc or {}).get("username") or user_id

    results = await loop.run_in_executor(None, lambda: faiss_store.similarity_search_with_score(processed, k=max_memories*3))
    mems = []
    for doc, score in results:
        md = doc.metadata
        item_id = md.get("item_id"); item_type = md.get("item_type")
        if not item_id or not item_type: continue
        col = conversations_col if item_type=="conversation" else journals_col
        id_field = "conversation_id" if item_type=="conversation" else "entry_id"
        q = {id_field:item_id, "user_id":[user_id] if item_type=="journal" else {"$in":[[speaker_id,user_id],[user_id,speaker_id]]}}
        base = await col.find_one(q)
        if not base: 
            await embeddings_col.delete_one({"item_id": item_id, "item_type": item_type})
            continue
        if item_type=="journal":
            base["speaker_name"] = target_name

        adjusted = 1.0 - score
        if item_type=="journal" and user_id in md.get("user_id", []): adjusted += 0.9
        elif md.get("speaker_id")==speaker_id or md.get("target_id")==user_id: adjusted += 0.7
        if speaker_name.lower() in base.get("content","").lower() or target_name.lower() in base.get("content","").lower():
            adjusted += 0.3
        ts = as_utc_aware(md.get("timestamp")) or as_utc_aware(base.get("timestamp"))
        days_old = (datetime.now(pytz.UTC) - ts).days if ts else 9999
        temporal_weight = 1/(1 + np.log1p(max(days_old,1)/30))
        adjusted *= temporal_weight
        if adjusted < 0.3: continue
        mems.append({
            "type": item_type, "content": base["content"], "timestamp": as_utc_aware(base["timestamp"]),
            "score": float(adjusted), "user_id": md.get("user_id", []),
            "speaker_id": md.get("speaker_id"), "speaker_name": base.get("speaker_name", target_name),
            "target_id": md.get("target_id"), "target_name": md.get("target_name")
        })
    mems.sort(key=lambda x: x["score"], reverse=True)
    return mems[:max_memories]

async def should_include_memories(user_input: str, speaker_id: str, user_id: str) -> Tuple[bool, List[dict]]:
    sp = await users_col.find_one({"user_id": speaker_id})
    speaker_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
    mems = await find_relevant_memories(speaker_id, user_id, user_input, speaker_name, max_memories=10)
    if not mems: return False, []
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(None, preprocess_input, user_input)
    inp = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
    rel = []
    for m in mems:
        emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(m["content"]))
        sim = np.dot(inp, emb) / (np.linalg.norm(inp)*np.linalg.norm(emb))
        if sim >= 0.5:
            rel.append(m)
    return (len(rel)>0), rel[:3]

async def initialize_bot(speaker_id: str, target_id: str, bot_role: str, user_input: str) -> Tuple[str,str,bool]:
    sp = await users_col.find_one({"user_id": speaker_id})
    tg = await users_col.find_one({"user_id": target_id})
    if not sp or not tg:
        raise ValueError("Invalid IDs")
    traits = await generate_personality_traits(target_id)
    recent = await get_recent_conversation_history(speaker_id, target_id)
    hist_text = "\n".join([f"{m['speaker']}: {m['content']}" for m in recent]) or "No recent conversation history."
    last_ts = recent[-1]["raw_timestamp"] if recent else None
    use_greeting = (not recent) or (datetime.now(pytz.UTC)-as_utc_aware(last_ts)).total_seconds()/60 > 30
    greeting, tone = await get_greeting_and_tone("friend" if not bot_role else bot_role, target_id)
    include, mems = await should_include_memories(user_input, speaker_id, target_id)
    mems_text = "No relevant memories."
    if include and mems:
        good = [m for m in mems if all(k in m for k in ["content","type","timestamp","speaker_name"])]
        if good:
            mems_text = "\n".join([f"- {m['content']} ({m['type']}, {m['timestamp'].strftime('%Y-%m-%d')}, said by {m['speaker_name']})" for m in good])
    if include:
        prompt = f"""
        You are {tg['display_name'] if 'display_name' in tg else tg['username']}, responding as an AI Twin to {sp.get('display_name', sp.get('username'))}, you are his/her {bot_role}.
        Generate a short 2-3 sentence reply that:
        - Uses a {tone} tone, appropriate for your relationship.
        - Reflects your personality: {', '.join([f"{k} ({v['explanation']})" for k,v in list(traits['core_traits'].items())[:3]])}.
        - Uses this recent context:
        {hist_text}
        - If relevant to '{user_input}', weave in one or two of these memories naturally:
        {mems_text}
        - {'Starts with "' + greeting + '" if no recent messages or time gap > 30 minutes.' if use_greeting else 'Do not start with a greeting.'}
        - Keep it short and personalized.
        Input: {user_input}
        """
    else:
        prompt = f"""
        You are {tg.get('display_name', tg.get('username'))}, responding as an AI Twin to {sp.get('display_name', sp.get('username'))}, his/her {bot_role}.
        Generate a short 2-3 sentence reply with a {tone} tone, reflecting your personality:
        {', '.join([f"{k} ({v['explanation']})" for k,v in list(traits['core_traits'].items())[:3]])}.
        Use this history:
        {hist_text}
        - {'Starts with "' + greeting + '" if no recent messages or time gap > 30 minutes.' if use_greeting else 'Do not start with a greeting.'}
        Input: {user_input}
        """
    return prompt, greeting, use_greeting

async def generate_response(prompt: str, user_input: str, greeting: str, use_greeting: bool) -> str:
    try:
        resp = await (await get_openai_client()).chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are an AI Twin responding in a personalized, casual manner."},
                {"role":"user","content":prompt}
            ],
            max_tokens=200, temperature=0.6
        )
        text = resp.choices[0].message.content.strip()
        if len(text.split()) >= 4 and ((use_greeting and text.lower().startswith(greeting.lower())) or not use_greeting):
            parts = text.split('. ')[:3]
            text = '. '.join([p for p in parts if p]).strip()
            if text and not text.endswith('.'): text += '.'
            return text
    except Exception as e:
        await errors_col.insert_one({"error": str(e), "input": user_input, "timestamp": datetime.now(pytz.UTC)})
    return f"{greeting}, sounds cool! What's up?" if use_greeting else "Sounds cool! What's up?"

# ---------------------
# Save message helper (used by HTTP & WS)
# ---------------------
async def save_and_embed_message(speaker_id: str, target_id: str, text: str, source: str) -> dict:
    await get_mongo_client()
    await ensure_faiss_store()
    sp = await users_col.find_one({"user_id": speaker_id})
    tg = await users_col.find_one({"user_id": target_id})
    sp_name = (sp or {}).get("display_name") or (sp or {}).get("username") or speaker_id
    tg_name = (tg or {}).get("display_name") or (tg or {}).get("username") or target_id
    now = datetime.now(pytz.UTC)
    conv_id = str(uuid.uuid4())
    doc = {
        "conversation_id": conv_id,
        "user_id": [speaker_id, target_id],
        "speaker_id": speaker_id,
        "speaker_name": sp_name,
        "target_id": target_id,
        "target_name": tg_name,
        "content": text,
        "type": "user_input" if source=="human" else "response",
        "source": source,
        "timestamp": now
    }
    await conversations_col.insert_one(doc)

    # embed & store
    processed = preprocess_input(text)
    loop = asyncio.get_event_loop()
    emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
    await embeddings_col.insert_one({
        "item_id": conv_id, "item_type":"conversation", "user_id":[speaker_id,target_id],
        "speaker_id": speaker_id, "speaker_name": sp_name,
        "target_id": target_id, "target_name": tg_name,
        "embedding": emb, "timestamp": now, "content": text
    })
    try:
        db_doc = Document(page_content=text, metadata={
            "item_id": conv_id, "item_type":"conversation", "user_id":[speaker_id,target_id],
            "speaker_id": speaker_id, "speaker_name": sp_name,
            "target_id": target_id, "target_name": tg_name,
            "timestamp": now
        })
        with faiss_lock:
            faiss_store.add_documents([db_doc])
            faiss_store.save_local(FAISS_DIR)
    except Exception as e:
        logger.warning(f"FAISS add fail: {e}")

    return doc

# ---------------------
# HTTP Chat (kept for compatibility)
# ---------------------
def require_api_and_session(sess=Depends(require_session), _: None = Depends(require_api_key)):
    return sess

@app.post("/send_message", response_model=MessageResponse)
async def send_message(req: MessageRequest, sess=Depends(require_api_and_session)):
    if sess["user"]["user_id"] != req.speaker_id:
        raise HTTPException(status_code=403, detail="Sender mismatch")
    await save_and_embed_message(req.speaker_id, req.target_id, req.user_input, source="human")
    # If target has AI enabled, generate immediate response
    tg = await users_col.find_one({"user_id": req.target_id})
    if tg and tg.get("ai_enabled", False):
        prompt, greeting, use_greeting = await initialize_bot(req.speaker_id, req.target_id, req.bot_role, req.user_input)
        ai_text = await generate_response(prompt, req.user_input, greeting, use_greeting)
        await save_and_embed_message(req.target_id, req.speaker_id, ai_text, source="ai_twin")
        return MessageResponse(response=ai_text)
    return MessageResponse(response="Sent.")

@app.get("/conversations/with/{other_id}")
async def history_with(other_id: str, limit: int = 30, sess=Depends(require_api_and_session)):
    me = sess["user"]["user_id"]
    cur = conversations_col.find({
        "user_id": {"$all":[me, other_id]}
    }).sort("timestamp",-1).limit(limit)
    out = []
    async for c in cur:
        out.append({
            "conversation_id": c["conversation_id"],
            "speaker_id": c["speaker_id"],
            "target_id": c["target_id"],
            "content": c["content"],
            "source": c.get("source","human"),
            "timestamp": as_utc_aware(c["timestamp"]).isoformat()
        })
    return {"messages": list(reversed(out))}

# ---------------------
# WebSocket Chat
# ---------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    user_id = websocket.query_params.get("user_id")
    # Basic guard for API key via cookie/header is omitted; socket is gated by session token
    await get_mongo_client()
    sess = await sessions_col.find_one({"token": token, "user_id": user_id})
    if not sess:
        await websocket.close(code=4401)
        return

    try:
        await manager.connect(user_id, websocket)
        await manager.broadcast_presence()
        await users_col.update_one({"user_id": user_id}, {"$set":{"last_seen": datetime.now(pytz.UTC)}})
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                continue
            if msg.get("type") == "chat":
                to = msg["to"]
                text = msg["text"]
                # persist sender message
                saved = await save_and_embed_message(user_id, to, text, source="human")
                # forward to recipient if online
                await manager.send_to(to, {"type":"chat","from": user_id, "payload":{
                    "speaker_id": saved["speaker_id"],
                    "target_id": saved["target_id"],
                    "content": saved["content"],
                    "source": "human",
                    "timestamp": saved["timestamp"].isoformat()
                }})
                # AI proxy?
                tgt = await users_col.find_one({"user_id": to})
                if tgt and tgt.get("ai_enabled", False):
                    prompt, greeting, use_greeting = await initialize_bot(user_id, to, "friend", text)
                    ai_text = await generate_response(prompt, text, greeting, use_greeting)
                    ai_saved = await save_and_embed_message(to, user_id, ai_text, source="ai_twin")
                    await manager.send_to(user_id, {"type":"ai","from": to, "payload":{
                        "speaker_id": ai_saved["speaker_id"],
                        "target_id": ai_saved["target_id"],
                        "content": ai_saved["content"],
                        "source": "ai_twin",
                        "timestamp": ai_saved["timestamp"].isoformat()
                    }})
            # (you can add typing indicators, read receipts, etc.)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(user_id)
        await manager.broadcast_presence()

# ---------------------
# Change streams (unchanged behavior, run concurrently)
# ---------------------
async def process_new_entry(item_id: str, item_type: str, content: str, user_id: list,
                            speaker_id: Optional[str] = None, speaker_name: Optional[str] = None,
                            target_id: Optional[str] = None, target_name: Optional[str] = None):
    try:
        await get_mongo_client()
        await ensure_faiss_store()
        processed = preprocess_input(content)
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(None, lambda: embeddings.embed_query(processed))
        now = datetime.now(pytz.UTC)
        doc = {
            "item_id": item_id, "item_type": item_type, "user_id": user_id,
            "content": content, "embedding": emb, "timestamp": now
        }
        if item_type=="conversation":
            doc.update({"speaker_id": speaker_id, "speaker_name": speaker_name, "target_id": target_id, "target_name": target_name})
        await embeddings_col.insert_one(doc)

        with faiss_lock:
            if faiss_store is None:
                faiss_store = FAISS.from_texts(["empty"], embeddings)
            meta = {"item_id": item_id, "item_type": item_type, "user_id": user_id, "timestamp": now}
            if item_type=="conversation":
                meta.update({"speaker_id": speaker_id, "speaker_name": speaker_name, "target_id": target_id, "target_name": target_name})
            faiss_store.add_documents([Document(page_content=content, metadata=meta)])
            faiss_store.save_local(FAISS_DIR)
    except Exception as e:
        await errors_col.insert_one({"error": str(e), "item_id": item_id, "item_type": item_type, "timestamp": datetime.now(pytz.UTC)})

async def watch_conversations():
    while True:
        try:
            await get_mongo_client()
            async with conversations_col.watch([{"$match":{"operationType":"insert"}}], full_document="updateLookup") as stream:
                async for change in stream:
                    doc = change["fullDocument"]
                    if doc.get("type") == "user_input" and doc.get("source") == "human":
                        await process_new_entry(
                            item_id=doc["conversation_id"], item_type="conversation",
                            content=doc["content"], user_id=doc["user_id"],
                            speaker_id=doc.get("speaker_id"), speaker_name=doc.get("speaker_name"),
                            target_id=doc.get("target_id"), target_name=doc.get("target_name")
                        )
        except Exception as e:
            await errors_col.insert_one({"error": str(e), "collection": "conversations", "timestamp": datetime.now(pytz.UTC)})
            await asyncio.sleep(5)

async def watch_journals():
    while True:
        try:
            await get_mongo_client()
            async with journals_col.watch([{"$match":{"operationType":"insert"}}], full_document="updateLookup") as stream:
                async for change in stream:
                    doc = change["fullDocument"]
                    await process_new_entry(item_id=doc["entry_id"], item_type="journal", content=doc["content"], user_id=doc["user_id"])
        except Exception as e:
            await errors_col.insert_one({"error": str(e), "collection": "journal_entries", "timestamp": datetime.now(pytz.UTC)})
            await asyncio.sleep(5)

async def watch_collections():
    await asyncio.gather(watch_conversations(), watch_journals())

# ---------------------
# Demo seed / initialization
# ---------------------
async def clear_database():
    await get_mongo_client()
    await users_col.delete_many({})
    await conversations_col.delete_many({})
    await journals_col.delete_many({})
    await embeddings_col.delete_many({})
    await relationships_col.delete_many({})
    await sessions_col.delete_many({})

async def populate_users():
    now = datetime.now(pytz.UTC)
    def mkuser(uid, uname, name):
        h=hash_password("password")
        return {"user_id": uid, "username": uname, "display_name": name, "password_salt":h["salt"], "password_hash":h["hash"], "ai_enabled": False, "created_at": now, "last_seen": now}
    base = [
        mkuser("user1","nipa","Nipa"),
        mkuser("user2","nick","Nick"),
        mkuser("user3","arif","Arif"),
        mkuser("user4","diana","Diana")
    ]
    for u in base:
        if not await users_col.find_one({"user_id": u["user_id"]}):
            await users_col.insert_one(u)

async def batch_embed_texts(texts: List[str]):
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: embeddings.embed_documents(texts))
    except Exception:
        return [None]*len(texts)

async def populate_conversations():
    now = datetime.now(pytz.UTC)
    convs = [
        {"conversation_id": str(uuid.uuid4()), "user_id":["user1","user2"], "speaker_id":"user1","speaker_name":"Nipa","target_id":"user2","target_name":"Nick","content":"Hey nick, ready for the project?","type":"user_input","source":"human","timestamp": now - timedelta(days=1)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user2","user1"], "speaker_id":"user2","speaker_name":"Nick","target_id":"user1","target_name":"Nipa","content":"Yeah, let's do this!","type":"user_input","source":"human","timestamp": now - timedelta(days=1, hours=1)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user3","user4"], "speaker_id":"user3","speaker_name":"Arif","target_id":"user4","target_name":"Diana","content":"Diana, got any weekend plans?","type":"user_input","source":"human","timestamp": now - timedelta(days=2)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user4","user3"], "speaker_id":"user4","speaker_name":"Diana","target_id":"user3","target_name":"Arif","content":"Just chilling, you?","type":"user_input","source":"human","timestamp": now - timedelta(days=2, hours=1)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user1","user3"], "speaker_id":"user1","speaker_name":"Nipa","target_id":"user3","target_name":"Arif","content":"Dad, I want to go to disney","type":"user_input","source":"human","timestamp": now - timedelta(hours=12)},
        {"conversation_id": str(uuid.uuid4()), "user_id":["user4","user2"], "speaker_id":"user4","speaker_name":"Diana","target_id":"user2","target_name":"Nick","content":"Nick, have you tried the new coffee shop yet?","type":"user_input","source":"human","timestamp": now - timedelta(hours=10)}
    ]
    for c in convs:
        if not await conversations_col.find_one({"conversation_id": c["conversation_id"]}):
            await conversations_col.insert_one(c)
    embeddings_result = await batch_embed_texts([c["content"] for c in convs])
    docs = []
    for c, e in zip(convs, embeddings_result):
        if e is not None and not await embeddings_col.find_one({"item_id": c["conversation_id"], "item_type":"conversation"}):
            docs.append({
                "item_id": c["conversation_id"], "item_type":"conversation", "user_id": c["user_id"],
                "content": c["content"], "embedding": e, "timestamp": c["timestamp"],
                "speaker_id": c["speaker_id"], "speaker_name": c["speaker_name"],
                "target_id": c["target_id"], "target_name": c["target_name"]
            })
    if docs: await embeddings_col.insert_many(docs)

async def populate_journals():
    now = datetime.now(pytz.UTC)
    j = {"entry_id": str(uuid.uuid4()), "user_id": ["user1"], "content":"I am in love with Jack", "timestamp": now - timedelta(hours=6)}
    if not await journals_col.find_one({"entry_id": j["entry_id"]}):
        await journals_col.insert_one(j)
    emb = (await batch_embed_texts([j["content"]]))[0]
    if emb is not None and not await embeddings_col.find_one({"item_id": j["entry_id"], "item_type":"journal"}):
        await embeddings_col.insert_one({"item_id": j["entry_id"], "item_type":"journal", "user_id": j["user_id"], "content": j["content"], "embedding": emb, "timestamp": j["timestamp"]})

async def verify_data():
    counts = {
        "Users": await users_col.count_documents({}),
        "Conversations": await conversations_col.count_documents({}),
        "Journals": await journals_col.count_documents({}),
        "Embeddings": await embeddings_col.count_documents({})
    }
    logger.info(f"DB counts: {counts}")

async def initialize_db():
    # dev/demo only: re-seed
    await clear_database()
    await populate_users()
    await populate_conversations()
    await populate_journals()
    await verify_data()
    await initialize_faiss_store()

# ---------------------
# Run
# ---------------------
# if __name__ == "__main__":
#     import uvicorn
#     # Set PUBLIC_UI_API_KEY in env and use it in the UI "x-api-key" box.
#     uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Use PORT from env, fallback to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
