# app.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, render_template_string
from embeddings_store import EmbStore
from logger import log_event
from llm import generate_answer  # OK if you haven't wired LLM; it will safely no-op

app = Flask(__name__)

# -------------------- Knowledge Base --------------------
KB = [
    {"q": "pre workout meal ideas running weightlifting gym breakfast",
     "a": ("Pre-workout (60–90 min): carbs + a little protein, low fat/fiber.\n"
           "Examples:\n• Oatmeal + banana + yogurt\n• Toast + peanut butter + fruit\n"
           "• Rice cake + turkey slices\nIf only 20–30 min: a small fruit (banana/applesauce).")},
    {"q": "post workout meal recovery what to eat after training",
     "a": ("Post-workout (within 1–2h): ~20–35g protein + carbs.\n"
           "Examples:\n• Greek yogurt + granola + berries\n• Chicken + rice + veggies\n"
           "• Protein shake + banana\n• Tofu stir-fry + noodles\nHydrate with water; add electrolytes if sweat is heavy.")},
    {"q": "how much protein do i need per day women female intake grams protein daily",
     "a": ("Most active adults: 1.2–1.6 g/kg body weight/day (up to 2.0 g/kg if heavy training).\n"
           "Example: 49 kg → ~60–80 g protein/day, spread across meals.")},
    {"q": "hydration plan water drink how much electrolytes weightlifting day",
     "a": ("Simple hydration plan:\n• Morning: 300–500 ml with breakfast\n• Pre-lift (1–2h): 300–500 ml\n"
           "• During: sip ~150–250 ml every 15–20 min\n• After: 300–500 ml; add electrolytes if session >60 min or sweat is heavy.\n"
           "Aim for pale-straw urine color.")},
    {"q": "20 minute home workout quick routine no equipment full body circuit",
     "a": ("20-minute circuit (no equipment):\n"
           "1) Squats 40s, Rest 20s\n2) Push-ups (knees OK) 40s, Rest 20s\n"
           "3) Glute bridges 40s, Rest 20s\n4) Plank 40s, Rest 20s\nRepeat 3 rounds.\n"
           "Lower impact: slow tempo. Harder: add a backpack for weight.")},
    {"q": "fat loss basics reduce body fat weight loss tips how to lose fat",
     "a": ("Fat-loss basics:\n1) Slight calorie deficit (~200–400 kcal/day)\n"
           "2) Protein 1.2–1.6 g/kg + fiber 25–35 g/day\n"
           "3) Train 2–3×/wk resistance + daily steps (7–10k)\nSleep 7–9h, manage stress, hydrate.")},
    {"q": "supplement timing protein creatine collagen when to take",
     "a": ("Protein: anytime; helpful post-workout or to hit daily target.\n"
           "Creatine: 3–5 g/day; timing doesn’t matter — take daily with water/food.\n"
           "Collagen: 10–15 g; pair with vitamin C for joints/skin.\nCheck personal tolerance and medical advice.")}
]

FALLBACK = ("I’m not sure yet. Try asking about: pre-workout, post-workout, protein needs, "
            "hydration, a 20-minute workout, fat-loss basics, or supplement timing.\n\n"
            "Note: Educational info only — not medical advice.")

# -------------------- Embeddings Store --------------------
store = EmbStore(KB)

def draft_answer(user_text: str):
    hits = store.search(user_text, k=3)
    if not hits:
        return None, []

    # confidence threshold (cosine similarity ~ 0..1)
    if hits[0]["score"] < 0.25:
        return None, hits

    # Compose grounded answer; include 1–2 sources as "References"
    top = hits[0]["a"]
    extra = ""
    if len(hits) > 1 and hits[1]["score"] > 0.2 and hits[1]["a"] not in top:
        extra = "\n\nAdditional tip:\n" + hits[1]["a"]

    refs = []
    for h in hits:
        if h["score"] > 0.2:
            refs.append(f"• Source {h['i']+1}: “{h['q'][:60]}…”")

    answer_txt = f"{top}{extra}\n\nReferences:\n" + ("\n".join(refs) if refs else "—")
    return answer_txt, hits

# -------------------- Guardrails --------------------
def should_refuse_medical(user_text: str) -> bool:
    t = user_text.lower()
    medical_terms = [
        "diagnose","diagnosis","medication","dose","mg","contraindication",
        "side effect","treat","treatment","prescription","prescribe"
    ]
    return any(w in t for w in medical_terms)

# Fitness domain check (keywords + semantic anchors)
FITNESS_KEYWORDS = {
    "workout","training","run","running","jog","jogging","cycle","cycling","bike",
    "strength","weights","weightlifting","lift","hiit","yoga","cardio","steps",
    "calorie","calories","nutrition","diet","protein","carb","carbs","fat","fiber",
    "hydration","water","electrolyte","electrolytes","creatine","collagen",
    "pre-workout","post-workout","warmup","cooldown","recovery","rest day","sleep",
    "meal","breakfast","lunch","dinner","snack","macros","deficit","surplus"
}
FITNESS_ANCHORS = [
    "fitness and nutrition advice",
    "workout programming and exercise tips",
    "hydration and electrolytes for training",
    "protein intake and recovery",
    "fat loss basics and macros",
    "home workouts and strength training"
]

def is_in_fitness_domain(user_text: str, model=None, sim_threshold: float = 0.22) -> bool:
    if not user_text.strip():
        return False
    t = user_text.lower()
    # fast keyword pass
    for kw in FITNESS_KEYWORDS:
        if kw in t:
            return True
    # semantic pass (reuse embeddings model)
    try:
        mdl = model or store.model
        q = mdl.encode([user_text], normalize_embeddings=True)
        anchors = mdl.encode(FITNESS_ANCHORS, normalize_embeddings=True)
        sims = (q @ anchors.T).ravel()
        return float(sims.max()) >= sim_threshold
    except Exception:
        return False

# -------------------- Chit-chat intent --------------------
STEER_BACK = "What would you like help with today — workouts or nutrition?"

CHITCHAT = {
    "how are you": "I’m doing great, thanks for asking!",
    "how's coach eva": "Coach FitEva here — always ready to help.",
    "who are you": "I’m Coach FitEva, your virtual fitness & nutrition coach (educational only).",
    "hello": "Hi there!",
    "hi": "Hey!"
}


def check_chitchat(user_text: str):
    t = user_text.lower()
    for k, v in CHITCHAT.items():
        if k in t:
            return v
    return None

# -------------------- LLM prompt (optional) --------------------
def build_grounded_prompt(user_text: str, hits) -> str:
    sources = []
    for i, h in enumerate(hits[:2]):
        sources.append(f"Source {i+1} (score={h['score']:.2f}):\n{h['a']}")
    sources_text = "\n\n".join(sources) if sources else "No sources."
    prompt = f"""
You are Coach FitEva. Answer the user's fitness/nutrition question using ONLY the sources below.
Be concise, friendly, and actionable. If the sources do not cover the request, say you don't know
and suggest what to ask instead. Do NOT include medical diagnosis or instructions.

User question:
{user_text}

Sources:
{sources_text}

Write the answer as short paragraphs or bullets. At the end, add:
References: Source 1{"/Source 2" if len(hits) > 1 else ""}

If you must refuse (not covered by sources), say:
"Sorry, I don't have that in my notes yet. Try asking about pre-workout, protein needs, hydration, a 20-minute workout, or supplement timing." Then add "References: —"
""".strip()
    return prompt

# -------------------- Brain: smart_reply --------------------
def smart_reply(user_text: str) -> str:

    cc = check_chitchat(user_text)
    if cc:
        log_event({"type": "chitchat", "q": user_text})
        return "Coach FitEva:\n" + cc + "\n\n" + STEER_BACK


    # 2) Medical guardrail
    if should_refuse_medical(user_text):
        return ("Coach FitEva:\nThanks for asking. I can’t help with medical diagnosis or specific medication guidance. "
                "For fitness/nutrition basics, ask me about pre-workout, protein needs, hydration, a 20-minute workout, or fat-loss fundamentals.\n\nReferences: —")

    # 3) Domain filter
    if not is_in_fitness_domain(user_text, model=store.model):
        log_event({"type": "out_of_scope", "q": user_text})
        return ("Coach FitEva:\nI’m focused on fitness & nutrition. "
                "Try asking about workouts, protein needs, hydration, recovery, or fat-loss basics.\n\nReferences: —")

    # 4) Retrieval (V2)
    out, hits = draft_answer(user_text)
    use_llm = os.getenv("USE_LLM") == "1"

    # 5) Optional LLM rewrite (V3)
    if out and use_llm:
        prompt = build_grounded_prompt(user_text, hits)
        llm_text = generate_answer(prompt)
        if llm_text:
            log_event({"type": "llm_answer", "q": user_text, "top_score": hits[0]["score"] if hits else None})
            return "Coach FitEva:\n" + llm_text

    if out:
        log_event({"type": "answer", "q": user_text, "top_score": hits[0]["score"] if hits else None})
        return "Coach FitEva:\n" + out

    # 6) Intent fallback (low-confidence retrieval)
    intents = {
        "protein":  "Try: ‘how much protein per day’ or ‘protein after workout’",
        "hydrate":  "Try: ‘daily hydration plan’ or ‘electrolytes when?’",
        "workout":  "Try: ‘20 minute home workout’ or ‘beginner strength plan’",
        "fat":      "Try: ‘fat loss basics’ or ‘steps + resistance split’",
    }
    low = user_text.lower()
    for k, v in intents.items():
        if k in low:
            msg = "Coach FitEva:\n" + v + "\n\n" + FALLBACK
            log_event({"type": "fallback_intent", "q": user_text, "intent": k})
            return msg

    log_event({"type": "fallback_generic", "q": user_text})
    return "Coach FitEva:\n" + FALLBACK

# -------------------- Web UI --------------------
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Fitness & Nutrition Chatbot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
   :root{
  --bg: #0f141a;          /* page background */
  --panel: #141a22;       /* cards/panels */
  --panel-2: #0f151d;     /* chat bubble (bot) */
  --panel-3: #1b2330;     /* chat bubble (me) */
  --border: #222a35;
  --text: #e6edf3;
  --muted: #9aa0a6;
  --input: #0f151d;
  --accent: #3b82f6;      /* buttons/links */
}
:root[data-theme='light']{
  --bg:#f7f9fc; --panel:#fff; --border:#d1d5db; --text:#111; --muted:#6b7280;
  /* chat bubbles */
  --bubble-me:#e0f2fe;         /* user bubble */
  --bubble-me-text:#111;
  --bubble-bot:#f3f4f6;        /* bot bubble */
  --bubble-bot-text:#111;
  /* buttons */
  --btn-bg:#2563eb; --btn-border:#2563eb; --btn-text:#fff; --btn-hover:#1d4ed8;
  --chip-bg:#f3f4f6; --chip-text:#374151; /* small feedback/quick-reply buttons */
}

:root[data-theme='dark']{
  --bg:#0f141a; --panel:#141a22; --border:#222a35; --text:#e6edf3; --muted:#9aa0a6;
  /* chat bubbles */
  --bubble-me:#1b2330;
  --bubble-me-text:#e6edf3;
  --bubble-bot:#0f151d;
  --bubble-bot-text:#e6edf3;
  /* buttons */
  --btn-bg:#3b82f6; --btn-border:#3b82f6; --btn-text:#fff; --btn-hover:#336fd1;
  --chip-bg:#0f151d; --chip-text:#cfd4d9;
}


*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--text);
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial
}
header{
  padding:16px 20px; background:var(--panel);
  border-bottom:1px solid var(--border)
}
.wrap{max-width:820px;margin:0 auto;padding:20px}

.card{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:16px; overflow:hidden;
  box-shadow:0 4px 12px rgba(0,0,0,.12);
}
/* Keep your base row flex */
.row{ display:flex; gap:12px; margin:8px 0; }

/* For bot rows: push bubble left, keep actions inline right */
.row.bot{ align-items:flex-start; }
.row.bot .bubble{ flex:1; }   /* bubble takes all available space */
.row.bot .actions{ margin-top:0; } /* keep aligned horizontally */

/* For user rows: bubble stays on the right */
.row.me{ justify-content:flex-end; }

/* Quick replies: force below bubble */
.row.bot .actions.quickreplies{
  flex-basis:100%; /* take whole row under bubble */
  margin-top:6px;
}


.bubble{
  max-width:72%; padding:12px 14px; border-radius:14px; line-height:1.45;
}
.bubble.me{
  background:var(--bubble-me);
  color:var(--bubble-me-text);
}
.bubble.bot{
  background:var(--bubble-bot);
  color:var(--bubble-bot-text);
  border:1px solid var(--border);
  white-space:pre-wrap;
}

.footer{
  display:flex; gap:10px; padding:12px;
  background:var(--panel); border-top:1px solid var(--border);
}
input[type=text]{
  flex:1; padding:12px 14px; border-radius:10px;
  border:1px solid var(--border);
  background:var(--panel); color:var(--text); outline:none;
}

button{
  padding:12px 16px; border-radius:10px;
  border:1px solid var(--btn-border);
  background:var(--btn-bg); color:var(--btn-text); cursor:pointer;
  transition: background .2s ease;
}
button:hover{ background:var(--btn-hover); }


.actions{ display:flex; gap:8px; margin-top:6px; }
.smallbtn{
  font-size:12px; padding:6px 8px; border-radius:8px;
  background:var(--chip-bg); border:1px solid var(--border);
  color:var(--chip-text); cursor:pointer;
}
.smallbtn{
  font-size:12px; padding:6px 10px; border-radius:8px;
  background:var(--chip-bg); border:1px solid var(--border);
  color:var(--chip-text); cursor:pointer;
  transition: background .2s ease, color .2s ease;
}
.smallbtn:hover{
  background:var(--btn-bg); 
  color:var(--btn-text); 
  border-color:var(--btn-border);
}
.smallbtn:disabled{ opacity:.6; cursor:default; }

/* Mode toggle base */
#themeToggle {
  font-size: 14px;
  padding: 6px 14px;
  border-radius: 8px;
  border: 1px solid #ccc;
  cursor: pointer;
  background: transparent;
  color: inherit;
  transition: background .2s ease, color .2s ease;
}

/* Hover in light mode */
:root[data-theme='light'] #themeToggle:hover {
  background: #000000;    /* white hover */
  color: #eaeaea;        /* text stays dark, visible */
  border-color: #ccc;
}

/* Hover in dark mode */
:root[data-theme='dark'] #themeToggle:hover {
  background: #ffffff;  /* black hover */
  color: #111111;        /* text stays light, visible */
  border-color: #444;
}





.chat{height:60vh;overflow:auto;padding:18px}
.row{display:flex;gap:12px;margin:8px 0}
.me{justify-content:flex-end}


.toast{
  position:fixed; bottom:20px; left:50%;
  transform:translateX(-50%) translateY(20px);
  background:var(--chip-bg); color:var(--text);
  padding:10px 14px; border:1px solid var(--border);
  border-radius:10px; box-shadow:0 6px 24px rgba(0,0,0,.15);
  opacity:0; transition:opacity .2s ease, transform .2s ease;
  pointer-events:none; font-size:14px; z-index:9999;
}
.toast.show{ opacity:1; transform:translateX(-50%) translateY(0); }


  </style>
</head>
<body data-theme="light">
  <header><div class="wrap"><strong>Fitness & Nutrition Chatbot</strong> <span class="hint">— demo (educational; not medical advice)</span>    <button id="themeToggle" class="smallbtn" style="float:right;">🌙 Dark Mode</button>
</div></header>
  <main class="wrap">
    <div class="card">
      <div id="chat" class="chat"></div>
      <div class="footer">
        <input id="msg" type="text" placeholder="Ask about pre-workout, hydration, protein…">
        <button id="send">Send</button>
      </div>
    </div>
    <p class="hint">Try: “what should I eat before running?”, “how much protein do I need?”, “20 minute home workout”.</p>
  </main>
  <div id="toast" class="toast" aria-live="polite" aria-atomic="true"></div>



<script>
const root = document.documentElement;
const toggleBtn = document.getElementById('themeToggle');

// Check if user had a preference saved
if(localStorage.getItem('theme')){
  root.setAttribute('data-theme', localStorage.getItem('theme'));
  toggleBtn.textContent = localStorage.getItem('theme') === 'light' ? "🌙 Dark Mode" : "☀️ Light Mode";
}

toggleBtn.addEventListener('click', ()=>{
  const current = root.getAttribute('data-theme');
  const next = current === 'light' ? 'dark' : 'light';
  root.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  toggleBtn.textContent = next === 'light' ? "🌙 Dark Mode" : "☀️ Light Mode";
});

const chat = document.getElementById('chat');
const msg = document.getElementById('msg');
const send = document.getElementById('send');
const toastEl = document.getElementById('toast');
let toastTimer = null;


function showToast(text){
  if(!toastEl) return;
  toastEl.textContent = text;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=> toastEl.classList.remove('show'), 1600);
}

function addBubble(text, who, options){
  const row = document.createElement('div');
  row.className = 'row ' + (who==='me' ? 'me' : 'bot');
  const b = document.createElement('div');
  b.className = 'bubble ' + (who==='me' ? 'me' : 'bot');
  b.textContent = text;
  row.appendChild(b);

  // Feedback buttons (bot messages only)
if (who !== 'me') {
  const actions = document.createElement('div');
  actions.className = 'actions';
  actions.innerHTML = `
    <button class="smallbtn" data-fb="up">👍 Helpful</button>
    <button class="smallbtn" data-fb="down">👎 Not helpful</button>
  `;
  actions.addEventListener('click', async (e)=>{
    const val = e.target.getAttribute('data-fb');
    if(!val) return;
    // disable both buttons after one click
    actions.querySelectorAll('button').forEach(b=> b.disabled = true);

    try{
      const r = await fetch('/api/feedback', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({q: window._lastUserQ || "", useful: (val==="up")})
      });
      const j = await r.json();
      if (j && j.ok){
        showToast(val === 'up' ? '✅ Feedback saved — thanks!' : '✅ Noted — we’ll improve this.');
      } else {
        showToast('⚠️ Could not save feedback.');
      }
    }catch(err){
      showToast('⚠️ Network error saving feedback.');
    }
  });
  row.appendChild(actions);
}


  // NEW: quick-reply buttons
  if (options && options.length > 0) {
    const qr = document.createElement('div');
qr.className = 'actions quickreplies';
    options.forEach(opt=>{
      const btn = document.createElement('button');
      btn.className = 'smallbtn';
      btn.textContent = opt;
      btn.addEventListener('click', ()=>{
        msg.value = opt;  // autofill input
        ask();            // auto-send
      });
      qr.appendChild(btn);
    });
    row.appendChild(qr);
  }

  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}


async function ask(){
  const text = msg.value.trim();
  if(!text) return;
  window._lastUserQ = text;
  addBubble(text,'me');
  msg.value='';
  const r = await fetch('/api/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:text})
  });
  const j = await r.json();
  addBubble(j.reply,'bot', j.options || []);
}


send.addEventListener('click', ask);
msg.addEventListener('keydown', (e)=>{ if(e.key==='Enter') ask(); });

addBubble("Hi! I’m Coach FitEva. Ask me about pre-workout, post-workout, protein, hydration, a 20-minute workout, fat-loss basics, or supplement timing.","bot");
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json(force=True)
    user_text = data.get("message","")
    reply = smart_reply(user_text)

    # If steer back is in reply, add quick-reply suggestions
    options = []
    if "workouts or nutrition" in reply.lower():
        options = ["Workouts", "Nutrition"]

    return jsonify({"reply": reply, "options": options})

@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(force=True)
    log_event({"type":"feedback", "q": data.get("q",""), "useful": bool(data.get("useful"))})
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True)
