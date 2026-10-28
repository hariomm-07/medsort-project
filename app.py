# Flask backend: classification, virtual sorting, bin monitoring, SQLite tracking.  Run: python app.py
import os, io, re, csv, math, sqlite3, colorsys
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory, abort, g, Response
from PIL import Image, UnidentifiedImageError
from waste_config import CLASSES, BIN_NAMES, STATUS_SORTED, STATUS_REVIEW
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED_STATIC = {".html", ".css", ".js", ".ico"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS records(
  id INTEGER PRIMARY KEY AUTOINCREMENT, submission_id TEXT UNIQUE,
  category TEXT NOT NULL, bin TEXT, class_id INTEGER, confidence REAL NOT NULL,
  created_at TEXT NOT NULL, status TEXT NOT NULL, collection_status TEXT NOT NULL,
  original_filename TEXT, source TEXT NOT NULL DEFAULT 'model');
CREATE TABLE IF NOT EXISTS bin_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, bin TEXT NOT NULL, event TEXT NOT NULL,
  created_at TEXT NOT NULL, after_record_id INTEGER NOT NULL, items_at_event INTEGER NOT NULL);
"""

def load_predictor(weights):
    """Return predict(PIL.Image)->list of 4 probabilities, or None if weights are missing/unusable."""
    if not os.path.exists(weights): return None
    try:
        import torch
        from torchvision import models, transforms
        m = models.mobilenet_v2(num_classes=len(CLASSES))
        m.load_state_dict(torch.load(weights, map_location="cpu")); m.eval()
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        def predict(img):
            with torch.no_grad(): return torch.softmax(m(tf(img).unsqueeze(0)), 1)[0].tolist()
        return predict
    except Exception as e:
        print("Model could not be loaded:", e); return None

def demo_predict(img):
    """DEMO ONLY - NOT AI. Guesses a bin from the average colour of the image. Opt-in via MEDSORT_DEMO=1."""
    r, g, b = img.resize((1, 1), Image.BILINEAR).getpixel((0, 0))
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if s < 0.25 and v > 0.55: i = 2                                  # whitish / grey  -> white
    elif h < 0.04 or h > 0.9: i = 1                                  # reddish         -> red
    elif 0.04 <= h < 0.25: i = 0                                     # yellow/orange   -> yellow
    elif 0.45 <= h <= 0.75: i = 3                                    # bluish          -> blue
    else: i = 0
    p = [0.05] * 4; p[i] = 0.85; return p

# ---- Zero-shot AI engine (OpenAI CLIP). Needs NO dataset, but is NOT trained or validated on medical waste. ----
CLIP_PROMPTS = {
    "yellow": ["a photo of soiled gauze", "a photo of a used bandage with blood", "a photo of cotton swabs and surgical dressings", "a photo of human tissue anatomical waste"],
    "red":    ["a photo of medical IV tubing", "a photo of a medical catheter", "a photo of a plastic syringe without a needle", "a photo of disposable medical gloves"],
    "white":  ["a photo of a hypodermic needle", "a photo of a scalpel blade", "a photo of medical sharps", "a photo of a used surgical blade"],
    "blue":   ["a photo of a glass medicine vial", "a photo of a glass ampoule", "a photo of a broken glass medicine bottle", "a photo of a metallic orthopedic implant"],
    "none":   ["a photo of an empty table", "a photo of a person", "a photo of a hand", "a photo of an ordinary household object"],   # 'nothing to sort' reject class
}

def aggregate_clip(logits, owner, n_classes=len(CLASSES) + 1):
    """Mean prompt logit per class -> softmax over 4 bins + 'none'. Returns only the 4 bin probabilities, so mass on
    'none' lowers the top confidence and the item goes to Manual Verification instead of being sorted."""
    means = []
    for k in range(n_classes):
        xs = [l for l, o in zip(logits, owner) if o == k]; means.append(sum(xs) / len(xs))
    m = max(means); ex = [math.exp(x - m) for x in means]; s = sum(ex)
    return [e / s for e in ex][:len(CLASSES)]

LAST_CLIP_ERROR = None

def load_clip_predictor():
    global LAST_CLIP_ERROR
    name = os.environ.get("MEDSORT_CLIP_MODEL", "openai/clip-vit-base-patch32")
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(name).eval(); proc = CLIPProcessor.from_pretrained(name)
    except Exception as e:
        LAST_CLIP_ERROR = f"{type(e).__name__}: {e}"[:300]; print("Zero-shot CLIP engine not available:", LAST_CLIP_ERROR); return None
    texts, owner = [], []
    for k, c in enumerate(CLASSES + ["none"]):
        for t in CLIP_PROMPTS[c]: texts.append(t); owner.append(k)
    def predict(img):
        with torch.no_grad():
            inp = proc(text=texts, images=img, return_tensors="pt", padding=True)
            logits = model(**inp).logits_per_image[0].tolist()
        return aggregate_clip(logits, owner)
    return predict

def err(code, msg, http):
    return jsonify(ok=False, error={"code": code, "message": msg}), http

def create_app(config=None):
    app = Flask(__name__, static_folder=None)
    app.config.update(
        DEMO=os.environ.get("MEDSORT_DEMO") == "1", IS_DEMO=False, IS_ZEROSHOT=False, ENGINE_ERROR=None, ENGINE=os.environ.get("MEDSORT_ENGINE", "auto"),
        DB_PATH=os.environ.get("MEDSORT_DB", os.path.join(HERE, "medsort_demo.db" if os.environ.get("MEDSORT_DEMO") == "1" else "medsort.db")),
        WEIGHTS=os.path.join(HERE, "waste_model.pt"),
        CONF_THRESHOLD=float(os.environ.get("CONF_THRESHOLD", 0.70)),
        BIN_CAPACITY=int(os.environ.get("BIN_CAPACITY", 20)),
        ALMOST_FULL=0.80, MIN_DIM=32, MAX_DIM=8000, MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        PREDICT=None, LOAD_MODEL=True)
    if config: app.config.update(config)
    if app.config["PREDICT"] is None and app.config["LOAD_MODEL"]:
        app.config["PREDICT"] = load_predictor(app.config["WEIGHTS"])
    if app.config["PREDICT"] is None and app.config["LOAD_MODEL"] and app.config["ENGINE"] in ("auto", "clip"):
        app.config["PREDICT"] = load_clip_predictor(); app.config["IS_ZEROSHOT"] = app.config["PREDICT"] is not None
        app.config["ENGINE_ERROR"] = None if app.config["PREDICT"] else LAST_CLIP_ERROR
    if app.config["PREDICT"] is None and app.config["DEMO"]:          # opt-in demo: separate DB, loudly labelled
        app.config["PREDICT"] = demo_predict; app.config["IS_DEMO"] = True
    if CORS: CORS(app)

    def db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DB_PATH"]); g.db.row_factory = sqlite3.Row
        return g.db
    @app.teardown_appcontext
    def close(_):
        d = g.pop("db", None)
        if d: d.close()
    with sqlite3.connect(app.config["DB_PATH"]) as c:                           # DB initialisation + migration
        c.executescript(SCHEMA)
        if "source" not in [r[1] for r in c.execute("PRAGMA table_info(records)")]:
            c.execute("ALTER TABLE records ADD COLUMN source TEXT NOT NULL DEFAULT 'model'")

    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    def rec(r): return {k: r[k] for k in r.keys()}

    def bin_state(name):
        cap = app.config["BIN_CAPACITY"]
        last = db().execute("SELECT MAX(after_record_id) m FROM bin_events WHERE bin=? AND event='emptied'", (name,)).fetchone()["m"] or 0
        n = db().execute("SELECT COUNT(*) n FROM records WHERE bin=? AND status=? AND id>?", (name, STATUS_SORTED, last)).fetchone()["n"]
        total = db().execute("SELECT COUNT(*) n FROM records WHERE bin=? AND status=?", (name, STATUS_SORTED)).fetchone()["n"]
        pct = min(100.0, round(n / cap * 100, 1))
        status = "Full" if n >= cap else "Almost Full" if n >= cap * app.config["ALMOST_FULL"] else "Normal"
        return {"bin": name, "label": BIN_NAMES[name], "class_id": CLASSES.index(name), "total_sorted": total,
                "items_since_emptied": n, "capacity": cap, "fullness_percent": pct, "status": status, "simulated": True}

    @app.route("/")
    def index(): return send_from_directory(HERE, "index.html")
    @app.route("/<path:f>")
    def static_files(f):
        if os.path.splitext(f)[1].lower() not in ALLOWED_STATIC: abort(404)
        return send_from_directory(HERE, f)

    def engine():
        return "demo" if app.config["IS_DEMO"] else "zeroshot" if app.config["IS_ZEROSHOT"] else "trained" if app.config["PREDICT"] else "none"

    @app.route("/api/health")
    def health():
        demo, eng = app.config["IS_DEMO"], engine()
        msgs = {"demo": "DEMO MODE: colour heuristic, not AI.", "zeroshot": "Zero-shot CLIP AI: not trained or validated on medical waste. Verify results.",
                "trained": None, "none": "No AI engine loaded (no waste_model.pt, no CLIP). Use manual selection."}
        return jsonify(ok=True, model_loaded=app.config["PREDICT"] is not None and not demo, demo_mode=demo, engine=eng, classes=CLASSES,
                       confidence_threshold=app.config["CONF_THRESHOLD"], bin_capacity=app.config["BIN_CAPACITY"], message=msgs[eng], engine_error=app.config["ENGINE_ERROR"] if eng == "none" else None)

    def read_image(f):
        try:
            img = Image.open(io.BytesIO(f.read()))
            if img.format not in ALLOWED_FORMATS: return None, err("bad_type", "Only JPEG, PNG, WEBP or BMP images are accepted.", 415)
            img.load(); img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            return None, err("bad_image", "File is not a valid or readable image.", 400)
        w, h = img.size
        if min(w, h) < app.config["MIN_DIM"] or max(w, h) > app.config["MAX_DIM"]:
            return None, err("bad_dimensions", f"Image must be between {app.config['MIN_DIM']} and {app.config['MAX_DIM']} px.", 400)
        return img, None

    @app.route("/api/detect", methods=["POST"])           # live preview: runs the AI but records NOTHING
    def detect():
        predict = app.config["PREDICT"]
        if predict is None: return err("model_unavailable", "No AI engine is loaded.", 503)
        f = request.files.get("image")
        if f is None: return err("no_image", "Send an image in the 'image' form field.", 400)
        img, e = read_image(f)
        if e: return e
        probs = predict(img)
        if len(probs) != len(CLASSES): return err("model_mismatch", "Model output does not match the 4 classes.", 500)
        i = max(range(len(probs)), key=lambda k: probs[k]); conf = float(probs[i])
        return jsonify(ok=True, category=CLASSES[i], class_id=i, confidence=conf, threshold=app.config["CONF_THRESHOLD"],
                       confident=conf >= app.config["CONF_THRESHOLD"], engine=engine(), recorded=False)

    @app.route("/api/classify", methods=["POST"])
    def classify():
        predict = app.config["PREDICT"]
        manual = (request.form.get("manual_category") or "").strip()      # operator-chosen bin (used when no trained model)
        if manual and manual not in CLASSES: return err("bad_category", "Unknown category.", 400)
        if predict is None and not manual: return err("model_unavailable", "No trained model (waste_model.pt) is loaded. Train one with train_model.py, or choose the bin manually.", 503)
        f = request.files.get("image")
        if f is None: return err("no_image", "Send an image in the 'image' form field.", 400)
        sid = (request.form.get("submission_id") or "").strip()
        if sid and not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", sid): return err("bad_submission_id", "Invalid submission_id.", 400)
        if sid:                                                  # duplicate submission: return stored result, no new count
            old = db().execute("SELECT * FROM records WHERE submission_id=?", (sid,)).fetchone()
            if old: return jsonify(ok=True, duplicate=True, **result_of(old))
        try:
            img = Image.open(io.BytesIO(f.read()))
            if img.format not in ALLOWED_FORMATS: return err("bad_type", "Only JPEG, PNG, WEBP or BMP images are accepted.", 415)
            img.load(); img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            return err("bad_image", "File is not a valid or readable image.", 400)
        w, h = img.size
        if min(w, h) < app.config["MIN_DIM"] or max(w, h) > app.config["MAX_DIM"]:
            return err("bad_dimensions", f"Image must be between {app.config['MIN_DIM']} and {app.config['MAX_DIM']} px.", 400)
        if manual:
            i, conf, source = CLASSES.index(manual), 1.0, "manual"
        else:
            probs = predict(img); source = "zeroshot" if app.config["IS_ZEROSHOT"] else "model"
            if len(probs) != len(CLASSES): return err("model_mismatch", "Model output does not match the 4 classes.", 500)
            i = max(range(len(probs)), key=lambda k: probs[k]); conf = float(probs[i])
        cat = CLASSES[i]
        ok = conf >= app.config["CONF_THRESHOLD"]
        name = os.path.basename(f.filename or "")[:120] or None
        try:
            cur = db().execute("INSERT INTO records(submission_id,category,bin,class_id,confidence,created_at,status,collection_status,original_filename,source) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (sid or None, cat, cat if ok else None, i if ok else None, conf, now(),
                 STATUS_SORTED if ok else STATUS_REVIEW, "Awaiting Collection" if ok else "Not Applicable", name, source))
            db().commit()
        except sqlite3.IntegrityError:                           # race on same submission_id
            old = db().execute("SELECT * FROM records WHERE submission_id=?", (sid,)).fetchone()
            return jsonify(ok=True, duplicate=True, **result_of(old))
        return jsonify(ok=True, duplicate=False, demo=app.config["IS_DEMO"], **result_of(db().execute("SELECT * FROM records WHERE id=?", (cur.lastrowid,)).fetchone()))

    def result_of(r):
        sorted_ = r["status"] == STATUS_SORTED
        return {"status": "sorted" if sorted_ else "manual_review", "record_id": r["id"], "category": r["category"],
                "class_id": r["class_id"], "bin": r["bin"], "confidence": r["confidence"], "source": r["source"], "threshold": app.config["CONF_THRESHOLD"],
                "message": f"Virtually sorted into the {BIN_NAMES[r['category']]} bin (simulation only{'; bin chosen manually, not by AI' if r['source']=='manual' else ''})." if sorted_
                           else "Manual Verification Required: confidence is below the threshold. Item was not counted."}

    def filtered(default_limit, max_limit):
        """Shared by /api/history and /api/export. Returns (rows, None) or (None, error_response)."""
        q, a = "SELECT * FROM records WHERE 1=1", []
        cat = request.args.get("category")
        if cat:
            if cat not in CLASSES: return None, err("bad_category", "Unknown category.", 400)
            q += " AND category=?"; a.append(cat)
        for key, op in (("date_from", ">="), ("date_to", "<=")):
            v = request.args.get(key)
            if v:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v): return None, err("bad_date", "Use YYYY-MM-DD.", 400)
                q += f" AND substr(created_at,1,10){op}?"; a.append(v)
        try: lim = max(1, min(max_limit, int(request.args.get("limit", default_limit))))
        except ValueError: return None, err("bad_limit", "limit must be an integer.", 400)
        return db().execute(q + " ORDER BY id DESC LIMIT ?", a + [lim]).fetchall(), None

    @app.route("/api/history")
    def history():
        rows, e = filtered(100, 500)
        return e if e else jsonify(ok=True, records=[rec(r) for r in rows])

    EXPORT_COLS = ["id", "created_at", "category", "bin", "confidence", "status", "collection_status", "source", "original_filename"]
    def csv_safe(v):                                   # stop spreadsheet formula injection from user-supplied file names
        if v is None: return ""
        if isinstance(v, float): return round(v, 4)
        v = str(v); return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v

    @app.route("/api/export")
    def export():
        rows, e = filtered(10000, 10000)
        if e: return e
        out = io.StringIO(); w = csv.writer(out); w.writerow(EXPORT_COLS)
        for r in rows: w.writerow([csv_safe(r[c]) for c in EXPORT_COLS])
        name = "medsort_records_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".csv"
        return Response("\ufeff" + out.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.route("/api/bins")
    def bins(): return jsonify(ok=True, bins=[bin_state(c) for c in CLASSES], note="Fullness is simulated; no sensors are connected.")

    @app.route("/api/bins/<name>/empty", methods=["POST"])
    def empty(name):
        if name not in CLASSES: return err("bad_bin", "Unknown bin.", 404)
        s = bin_state(name)
        last = db().execute("SELECT COALESCE(MAX(id),0) m FROM records").fetchone()["m"]
        db().execute("INSERT INTO bin_events(bin,event,created_at,after_record_id,items_at_event) VALUES(?,?,?,?,?)",
                     (name, "emptied", now(), last, s["items_since_emptied"])); db().commit()
        return jsonify(ok=True, bin=bin_state(name), message=f"{BIN_NAMES[name]} bin marked as emptied (record only; disposal is not confirmed).")

    @app.route("/api/bin-events")
    def bin_events():
        rows = db().execute("SELECT * FROM bin_events ORDER BY id DESC LIMIT 100").fetchall()
        return jsonify(ok=True, events=[rec(r) for r in rows])

    @app.route("/api/summary")
    def summary():
        d = db()
        by = {r["category"]: r["n"] for r in d.execute("SELECT category, COUNT(*) n FROM records WHERE status=? GROUP BY category", (STATUS_SORTED,))}
        return jsonify(ok=True, total_records=d.execute("SELECT COUNT(*) n FROM records").fetchone()["n"],
            total_sorted=sum(by.values()), manual_review=d.execute("SELECT COUNT(*) n FROM records WHERE status=?", (STATUS_REVIEW,)).fetchone()["n"],
            sorted_by_category={c: by.get(c, 0) for c in CLASSES}, emptying_events=d.execute("SELECT COUNT(*) n FROM bin_events").fetchone()["n"],
            disposal_confirmed=0, note="Disposal Not Confirmed: no collection or disposal data exists.")

    @app.route("/api/state")                                       # kept for backward compatibility
    def state():
        s = summary().get_json(); return jsonify(counts=s["sorted_by_category"], total=s["total_sorted"])

    @app.errorhandler(413)
    def too_big(_): return err("too_large", "Image is larger than 8 MB.", 413)
    return app

app = create_app()

if __name__ == "__main__":
    a = create_app(); c = a.config
    eng = "demo (colour heuristic, NOT AI)" if c["IS_DEMO"] else "zero-shot CLIP (not validated on medical waste)" if c["IS_ZEROSHOT"] else "trained model waste_model.pt" if c["PREDICT"] else "NONE - manual selection only"
    print("\n==== MedSort ====\nAI engine:", eng)
    if not c["PREDICT"]:
        print("To enable automatic detection run setup_ai.bat (or: pip install torch transformers) and restart.", ("\nReason: " + c["ENGINE_ERROR"]) if c["ENGINE_ERROR"] else "")
    print("Open http://127.0.0.1:5000\n")
    a.run(debug=True, use_reloader=False)               # reloader off: it would load the AI model twice
