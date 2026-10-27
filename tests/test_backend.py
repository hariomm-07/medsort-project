import io, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from app import create_app
from waste_config import CLASSES

def png(size=(64, 64), color=(200, 30, 30)):
    b = io.BytesIO(); Image.new("RGB", size, color).save(b, "PNG"); b.seek(0); return b

class Base(unittest.TestCase):
    force = (0, 0.95)                                   # (class index, confidence) returned by the fake model
    def predict(self, img):
        i, c = self.force; p = [(1 - c) / 3] * 4; p[i] = c; return p
    def make(self, predict="default", **cfg):
        self.db = os.path.join(self.tmp.name, "t.db")
        self.app = create_app(dict(DB_PATH=self.db, LOAD_MODEL=False, PREDICT=self.predict if predict == "default" else predict, **cfg))
        self.c = self.app.test_client()
    def setUp(self): self.tmp = tempfile.TemporaryDirectory(); self.make()
    def tearDown(self): self.tmp.cleanup()
    def up(self, data=None, name="x.png", sid=None):
        d = {"image": (data or png(), name)}
        if sid: d["submission_id"] = sid
        return self.c.post("/api/classify", data=d, content_type="multipart/form-data")
    def bins(self): return {b["bin"]: b for b in self.c.get("/api/bins").get_json()["bins"]}

class T(Base):
    def test_01_health(self):
        j = self.c.get("/api/health").get_json(); self.assertTrue(j["ok"]); self.assertTrue(j["model_loaded"]); self.assertEqual(j["classes"], CLASSES)
    def test_02_upload_inference(self):
        r = self.up(); j = r.get_json(); self.assertEqual(r.status_code, 200); self.assertEqual(j["status"], "sorted"); self.assertEqual(j["bin"], "yellow")
    def test_03_invalid_type(self):
        r = self.up(io.BytesIO(b"hello text"), "a.png"); self.assertEqual(r.status_code, 400)
        b = io.BytesIO(); Image.new("RGB", (64, 64)).save(b, "GIF"); b.seek(0); self.assertEqual(self.up(b, "a.gif").status_code, 415)
    def test_04_corrupted_image(self):
        good = png().read(); r = self.up(io.BytesIO(good[:40]), "c.png"); self.assertEqual(r.status_code, 400); self.assertEqual(r.get_json()["error"]["code"], "bad_image")
    def test_04b_dimensions(self): self.assertEqual(self.up(png((8, 8))).get_json()["error"]["code"], "bad_dimensions")
    def test_05_missing_weights(self):
        self.make(predict=None); r = self.up(); self.assertEqual(r.status_code, 503); self.assertEqual(r.get_json()["error"]["code"], "model_unavailable")
        self.assertFalse(self.c.get("/api/health").get_json()["model_loaded"]); self.assertEqual(self.c.get("/api/summary").get_json()["total_records"], 0)
    def test_06_low_confidence(self):
        self.force = (1, 0.40); j = self.up().get_json(); self.assertEqual(j["status"], "manual_review"); self.assertIsNone(j["bin"])
        self.assertIn("Manual Verification Required", j["message"])
    def test_07_mapping_all_classes(self):
        for i, c in enumerate(CLASSES):
            self.force = (i, 0.9); j = self.up().get_json(); self.assertEqual((j["class_id"], j["bin"]), (i, c))
    def test_08_count_once_and_duplicates(self):
        self.up(sid="sub-00000001"); j = self.up(sid="sub-00000001").get_json(); self.assertTrue(j["duplicate"])
        self.assertEqual(self.bins()["yellow"]["total_sorted"], 1); self.assertEqual(self.c.get("/api/summary").get_json()["total_records"], 1)
    def test_09_uncertain_not_counted(self):
        self.force = (2, 0.3); self.up(); self.assertEqual(sum(b["total_sorted"] for b in self.bins().values()), 0)
        s = self.c.get("/api/summary").get_json(); self.assertEqual((s["manual_review"], s["total_sorted"]), (1, 0))
    def test_10_history_persists_restart(self):
        self.up(name="keep.png"); self.make()               # new app object, same DB file = restart
        h = self.c.get("/api/history").get_json()["records"]; self.assertEqual(len(h), 1); self.assertEqual(h[0]["original_filename"], "keep.png")
        self.assertEqual(h[0]["collection_status"], "Awaiting Collection")
    def test_11_fullness_and_capacity(self):
        self.make(BIN_CAPACITY=5); self.up(); b = self.bins()["yellow"]; self.assertEqual((b["fullness_percent"], b["status"]), (20.0, "Normal"))
        for _ in range(3): self.up()
        self.assertEqual(self.bins()["yellow"]["status"], "Almost Full")          # 4/5 = 80%
        self.up(); self.up(); b = self.bins()["yellow"]                            # 6th item exceeds capacity
        self.assertEqual((b["fullness_percent"], b["status"]), (100.0, "Full"))
    def test_12_empty_preserves_history(self):
        self.make(BIN_CAPACITY=4); self.up(); self.up()
        r = self.c.post("/api/bins/yellow/empty").get_json(); self.assertEqual(r["bin"]["fullness_percent"], 0.0)
        b = self.bins()["yellow"]; self.assertEqual((b["total_sorted"], b["items_since_emptied"]), (2, 0))
        self.assertEqual(len(self.c.get("/api/history").get_json()["records"]), 2); self.assertEqual(len(self.c.get("/api/bin-events").get_json()["events"]), 1)
        self.up(); self.assertEqual(self.bins()["yellow"]["items_since_emptied"], 1)
    def test_13_untrusted_input(self):
        self.assertEqual(self.c.post("/api/bins/evil'--/empty").status_code, 404)
        self.assertEqual(self.c.get("/api/history?category=x' OR 1=1").status_code, 400)
        self.assertEqual(self.c.get("/api/history?date_from=bad").status_code, 400)
    def test_14_filters(self):
        self.force = (0, .9); self.up(); self.force = (3, .9); self.up()
        self.assertEqual(len(self.c.get("/api/history?category=blue").get_json()["records"]), 1)
    def test_15_source_not_served(self):
        self.assertEqual(self.c.get("/app.py").status_code, 404); self.assertEqual(self.c.get("/medsort.db").status_code, 404)
class Manual(Base):
    def post(self, cat, sid=None):
        d = {"image": (png(), "x.png"), "manual_category": cat}
        if sid: d["submission_id"] = sid
        return self.c.post("/api/classify", data=d, content_type="multipart/form-data")
    def test_manual_works_without_model(self):
        self.make(predict=None); j = self.post("red").get_json(); self.assertEqual((j["status"], j["bin"], j["source"]), ("sorted", "red", "manual"))
        self.assertEqual(self.bins()["red"]["total_sorted"], 1); self.assertIn("manually", j["message"])
    def test_manual_rejects_bad_category_and_bad_image(self):
        self.make(predict=None); self.assertEqual(self.post("evil").status_code, 400)
        r = self.c.post("/api/classify", data={"image": (io.BytesIO(b"nope"), "a.png"), "manual_category": "red"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400); self.assertEqual(self.bins()["red"]["total_sorted"], 0)
    def test_manual_duplicate_counts_once(self):
        self.make(predict=None); self.post("blue", "dup-0000001"); self.assertTrue(self.post("blue", "dup-0000001").get_json()["duplicate"]); self.assertEqual(self.bins()["blue"]["total_sorted"], 1)
    def test_old_database_is_migrated(self):
        import sqlite3; old = os.path.join(self.tmp.name, "old.db"); c = sqlite3.connect(old)
        c.executescript("CREATE TABLE records(id INTEGER PRIMARY KEY AUTOINCREMENT, submission_id TEXT UNIQUE, category TEXT NOT NULL, bin TEXT, class_id INTEGER, confidence REAL NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, collection_status TEXT NOT NULL, original_filename TEXT);"); c.close()
        a = create_app(dict(DB_PATH=old, LOAD_MODEL=False, PREDICT=None)).test_client()
        self.assertEqual(a.get("/api/history").status_code, 200)
        self.assertEqual(a.post("/api/classify", data={"image": (png(), "x.png"), "manual_category": "red"}, content_type="multipart/form-data").status_code, 200)
class Export(Base):
    def rows(self, qs=""):
        import csv as _c; r = self.c.get("/api/export" + qs); txt = r.get_data(as_text=True).lstrip("\ufeff")
        return r, list(_c.reader(io.StringIO(txt)))
    def test_export_csv_columns_and_rows(self):
        self.up(name="a.png"); self.force = (3, .9); self.up(name="b.png")
        r, rows = self.rows(); self.assertEqual(r.status_code, 200); self.assertIn("text/csv", r.content_type); self.assertIn("attachment", r.headers["Content-Disposition"])
        self.assertEqual(rows[0], ["id", "created_at", "category", "bin", "confidence", "status", "collection_status", "source", "original_filename"]); self.assertEqual(len(rows), 3)
    def test_export_respects_filters(self):
        self.up(); self.force = (3, .9); self.up(); _, rows = self.rows("?category=blue"); self.assertEqual([r[2] for r in rows[1:]], ["blue"])
    def test_export_includes_manual_review_rows(self):
        self.force = (1, .3); self.up(); _, rows = self.rows(); self.assertEqual(rows[1][5], "Manual Verification Required")
    def test_export_blocks_formula_injection(self):
        self.up(name="=HYPERLINK(1).png"); _, rows = self.rows(); self.assertTrue(rows[1][8].startswith("'="))
    def test_export_bad_filters(self):
        self.assertEqual(self.c.get("/api/export?category=x").status_code, 400); self.assertEqual(self.c.get("/api/export?date_from=zz").status_code, 400)
    def test_export_empty_has_header_only(self):
        _, rows = self.rows(); self.assertEqual(len(rows), 1)
class Detect(Base):
    def det(self, data=None):
        return self.c.post("/api/detect", data={"image": (data or png(), "f.jpg")}, content_type="multipart/form-data")
    def test_detect_does_not_record(self):
        j = self.det().get_json(); self.assertTrue(j["confident"]); self.assertFalse(j["recorded"]); self.assertEqual(j["category"], "yellow")
        self.assertEqual(self.c.get("/api/summary").get_json()["total_records"], 0)
    def test_detect_low_confidence_not_confident(self):
        self.force = (2, 0.4); self.assertFalse(self.det().get_json()["confident"])
    def test_detect_errors(self):
        self.assertEqual(self.det(io.BytesIO(b"junk")).status_code, 400)
        self.make(predict=None); self.assertEqual(self.det().status_code, 503)
    def test_engine_reported_in_health(self):
        self.assertEqual(self.c.get("/api/health").get_json()["engine"], "trained")
        self.make(IS_ZEROSHOT=True); h = self.c.get("/api/health").get_json(); self.assertEqual(h["engine"], "zeroshot"); self.assertTrue(h["model_loaded"])
        self.make(predict=None); self.assertEqual(self.c.get("/api/health").get_json()["engine"], "none")
    def test_zeroshot_source_recorded(self):
        self.make(IS_ZEROSHOT=True); self.assertEqual(self.up().get_json()["source"], "zeroshot")
class Engine(Base):
    def test_health_explains_missing_engine(self):
        try: import transformers; self.skipTest("transformers installed")
        except ImportError: pass
        a = create_app(dict(DB_PATH=os.path.join(self.tmp.name, "e.db"), WEIGHTS=os.path.join(self.tmp.name, "none.pt"))).test_client()
        h = a.get("/api/health").get_json(); self.assertEqual(h["engine"], "none"); self.assertIn("No module named", h["engine_error"])
    def test_no_engine_error_when_engine_present(self):
        self.assertIsNone(self.c.get("/api/health").get_json()["engine_error"])
class Clip(unittest.TestCase):
    def test_aggregate_picks_class_and_rejects_none(self):
        from app import aggregate_clip, CLIP_PROMPTS
        owner = [k for k, c in enumerate(CLASSES + ["none"]) for _ in CLIP_PROMPTS[c]]
        hot = lambda k: [30.0 if o == k else 20.0 for o in owner]
        p = aggregate_clip(hot(2), owner); self.assertEqual(p.index(max(p)), 2); self.assertGreater(max(p), 0.9); self.assertEqual(len(p), 4)
        q = aggregate_clip(hot(4), owner); self.assertLess(max(q), 0.05)           # 'none' wins -> every bin probability is tiny
    def test_clip_unavailable_returns_none_without_transformers(self):
        from app import load_clip_predictor
        try: import transformers; self.skipTest("transformers installed")
        except ImportError: self.assertIsNone(load_clip_predictor())
if __name__ == "__main__": unittest.main()
