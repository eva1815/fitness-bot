from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---- Domain knowledge (edit freely) ----
KB = [
    {
        "q": "pre workout meal ideas running weightlifting gym breakfast",
        "a": (
            "Pre-workout (60–90 min): carbs + a little protein, low fat/fiber.\n"
            "Examples:\n• Oatmeal + banana + yogurt\n• Toast + peanut butter + fruit\n"
            "• Rice cake + turkey slices\nIf only 20–30 min: a small fruit (banana/applesauce)."
        )
    },
    {
        "q": "post workout meal recovery what to eat after training",
        "a": (
            "Post-workout (within 1–2h): ~20–35g protein + carbs.\n"
            "Examples:\n• Greek yogurt + granola + berries\n• Chicken + rice + veggies\n"
            "• Protein shake + banana\n• Tofu stir-fry + noodles\nHydrate with water; add electrolytes if sweat is heavy."
        )
    },
    {
        "q": "how much protein do i need per day women female intake grams protein daily",
        "a": (
            "Most active adults: 1.2–1.6 g/kg body weight/day (up to 2.0 g/kg if heavy training).\n"
            "Example: 49 kg → ~60–80 g protein/day, spread across meals."
        )
    },
    {
        "q": "hydration plan water drink how much electrolytes weightlifting day",
        "a": (
            "Simple hydration plan:\n• Morning: 300–500 ml with breakfast\n• Pre-lift (1–2h): 300–500 ml\n"
            "• During: sip ~150–250 ml every 15–20 min\n• After: 300–500 ml; add electrolytes if session >60 min or sweat is heavy.\n"
            "Aim for pale-straw urine color."
        )
    },
    {
        "q": "20 minute home workout quick routine no equipment full body circuit",
        "a": (
            "20-minute circuit (no equipment):\n"
            "1) Squats 40s, Rest 20s\n2) Push-ups (knees OK) 40s, Rest 20s\n"
            "3) Glute bridges 40s, Rest 20s\n4) Plank 40s, Rest 20s\nRepeat 3 rounds.\n"
            "Lower impact: slow tempo. Harder: add a backpack for weight."
        )
    },
    {
        "q": "fat loss basics reduce body fat weight loss tips how to lose fat",
        "a": (
            "Fat-loss basics:\n1) Slight calorie deficit (~200–400 kcal/day)\n"
            "2) Protein 1.2–1.6 g/kg + fiber 25–35 g/day\n"
            "3) Train 2–3×/wk resistance + daily steps (7–10k)\nSleep 7–9h, manage stress, hydrate."
        )
    },
    {
        "q": "supplement timing protein creatine collagen when to take",
        "a": (
            "Protein: anytime; helpful post-workout or to hit daily target.\n"
            "Creatine: 3–5 g/day; timing doesn’t matter — take daily with water/food.\n"
            "Collagen: 10–15 g; pair with vitamin C for joints/skin.\nCheck personal tolerance and medical advice."
        )
    },
]

FALLBACK = (
    "I’m not sure yet. Try asking about: pre-workout, post-workout, protein needs, "
    "hydration, a 20-minute workout, fat-loss basics, or supplement timing.\n\n"
    "Note: Educational info only — not medical advice."
)

# Build a TF-IDF index over KB questions
vectorizer = TfidfVectorizer(stop_words="english")
kb_questions = [item["q"] for item in KB]
kb_matrix = vectorizer.fit_transform(kb_questions)

def answer(user_text: str) -> str:
    """Return the best answer from KB using cosine similarity; fallback below threshold."""
    if not user_text.strip():
        return FALLBACK
    q_vec = vectorizer.transform([user_text])
    sims = cosine_similarity(q_vec, kb_matrix)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score < 0.15:  # threshold: tweak as you grow the KB
        return FALLBACK
    return KB[best_idx]["a"]

# -------------- Web UI --------------

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Fitness & Nutrition Chatbot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --bg:#0b0c10; --panel:#111318; --text:#eaeaea; --muted:#9aa0a6; }
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial}
    header{padding:16px 20px;background:#16181d;border-bottom:1px solid #1f232b}
    .wrap{max-width:820px;margin:0 auto;padding:20px}
    .card{background:var(--panel);border:1px solid #1f232b;border-radius:16px;overflow:hidden;box-shadow:0 6px 24px rgba(0,0,0,.25)}
    .chat{height:60vh;overflow:auto;padding:18px}
    .row{display:flex;gap:12px;margin:8px 0}
    .me{justify-content:flex-end}
    .bubble{max-width:72%;padding:12px 14px;border-radius:14px;line-height:1.45}
    .bubble.me{background:#2a2f3a}
    .bubble.bot{background:#1a1f27;border:1px solid #2a2f3a}
    .footer{display:flex;gap:10px;border-top:1px solid #1f232b;padding:12px;background:#14171c}
    input[type=text]{flex:1;padding:12px 14px;border-radius:10px;border:1px solid #2a2f3a;background:#0f1217;color:var(--text);outline:none}
    button{padding:12px 16px;border-radius:10px;border:1px solid #2a2f3a;background:#2a2f3a;color:#fff;cursor:pointer}
    small{color:var_muted}
    .hint{color:var(--muted);font-size:14px;margin-top:8px}
    a{color:#81c7ff}
  </style>
</head>
<body>
  <header><div class="wrap"><strong>Fitness & Nutrition Chatbot</strong> <span class="hint">— demo project (educational; not medical advice)</span></div></header>
  <main class="wrap">
    <div class="card">
      <div id="chat" class="chat"></div>
      <div class="footer">
        <input id="msg" type="text" placeholder="Ask me about pre-workout, hydration, protein…">
        <button id="send">Send</button>
      </div>
    </div>
    <p class="hint">Try: “what should I eat before running?”, “how much protein do I need?”, “20 minute home workout”.</p>
  </main>
<script>
const chat = document.getElementById('chat');
const msg = document.getElementById('msg');
const send = document.getElementById('send');

function addBubble(text, who){
  const row = document.createElement('div');
  row.className = 'row ' + (who==='me' ? 'me' : 'bot');
  const b = document.createElement('div');
  b.className = 'bubble ' + (who==='me' ? 'me' : 'bot');
  b.textContent = text;
  row.appendChild(b);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

async function ask(){
  const text = msg.value.trim();
  if(!text) return;
  addBubble(text,'me');
  msg.value='';
  const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:text})});
  const j = await r.json();
  addBubble(j.reply,'bot');
}

send.addEventListener('click', ask);
msg.addEventListener('keydown', (e)=>{ if(e.key==='Enter') ask(); });

addBubble("Hi! I can help with quick fitness & nutrition tips. Ask me about pre-workout, post-workout, protein, hydration, a 20-minute workout, fat-loss basics, or supplement timing.","bot");
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
    return jsonify({"reply": answer(user_text)})

if __name__ == "__main__":
    app.run(debug=True)
