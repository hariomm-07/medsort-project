# MedSort: Smart Medical-Waste Segregation (software prototype)

Software simulation only. No hardware, no physical sorting, no confirmed collection or disposal.

## Folder structure
```
medsort/
  index.html  style.css  app.js        website (served by Flask)
  app.py                               Flask API + SQLite (medsort.db is created automatically)
  waste_config.py                      class order, shared by training/backend/frontend
  train_model.py                       trains MobileNetV2 -> waste_model.pt, metrics.json
  requirements.txt
  tests/test_backend.py                automated tests (unittest)
  dataset/{train,val,test}/{yellow,red,white,blue}/   your labelled images
```
Class order everywhere: `yellow=0, red=1, white=2, blue=3`.

## Windows 11 PowerShell
```powershell
cd path\to\medsort
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python train_model.py --epochs 15      # needs images, see below; creates waste_model.pt + metrics.json
python app.py                          # open http://127.0.0.1:5000
python -m unittest discover -s tests -v
```
Optional settings: `$env:CONF_THRESHOLD="0.75"; $env:BIN_CAPACITY="30"; python app.py`

## Prepare the dataset
1. Put each image in the folder of its bin: `yellow` (soiled dressings, gauze), `red` (tubing, catheters, needle-less syringes), `white` (needles, scalpels, sharps), `blue` (glass vials, ampoules, metal implants).
2. Aim for at least 150-300 images per class. Use real photos with varied backgrounds, angles and lighting, not only web images of one style.
3. Split by **item/photo session**, not randomly by file, so near-duplicates do not appear in both train and test: about 70% `train`, 15% `val`, 15% `test`.
4. Keep `test` untouched until the end. `train_model.py` reports accuracy, precision, recall, F1 and the confusion matrix on `test` (falls back to `val` and says so).
5. Have a qualified person confirm labels against your local biomedical-waste rules. The model learns your labels, nothing more.
6. No images, no model: the site then shows "classification unavailable" and refuses to classify.

## API
All JSON responses contain `ok`. Errors: `{"ok":false,"error":{"code":"...","message":"..."}}`.
```powershell
curl.exe http://127.0.0.1:5000/api/health
curl.exe -X POST -F "image=@needle.jpg" -F "submission_id=abc12345-xyz" http://127.0.0.1:5000/api/classify
curl.exe "http://127.0.0.1:5000/api/history?category=blue&date_from=2026-10-01"
curl.exe http://127.0.0.1:5000/api/bins
curl.exe -X POST http://127.0.0.1:5000/api/bins/yellow/empty
curl.exe http://127.0.0.1:5000/api/summary
```
`/api/classify` success:
```json
{"ok":true,"duplicate":false,"status":"sorted","record_id":1,"category":"white","class_id":2,"bin":"white","confidence":0.93,"threshold":0.7,"message":"Virtually sorted into the White translucent bin (simulation only)."}
```
Low confidence: `"status":"manual_review","bin":null` and the item is not counted. No model: HTTP 503 `model_unavailable`. Bad file: 400/415 (`bad_image`, `bad_type`, `bad_dimensions`).
`/api/bins` item: `{"bin":"yellow","total_sorted":5,"items_since_emptied":5,"capacity":20,"fullness_percent":25.0,"status":"Normal","simulated":true}`

## Manual test plan (not automatable here)
- Stop the server, press Classify: UI shows "Cannot reach the backend", no crash.
- Without `waste_model.pt`: banner says unavailable; Classify gives the unavailable message.
- Upload a .txt renamed .jpg: error shown, nothing counted.
- Press Classify twice quickly: only one record and one count.
- Resize to 360 px wide: menu, bins, tables and cards fit without sideways scroll.

## Camera and CSV download
- "Use camera" opens the webcam (works on http://localhost or https). Show the item, press Capture photo, then Classify & sort.
- "Download CSV" in the Tracking records section exports the records (respecting the category/date filters): `curl.exe -o records.csv "http://127.0.0.1:5000/api/export?category=blue&date_from=2026-10-01"`. Open in Excel; columns: id, created_at (UTC), category, bin, confidence, status, collection_status, source (model/manual), original_filename.

## AI engines (auto-selected at start-up, in this order)
1. **Trained** - `waste_model.pt` from `train_model.py` (best; accuracy measured on your test set).
2. **Zero-shot CLIP** - `pip install transformers` (downloads ~600 MB once). Needs no dataset, matches the image to text descriptions of each bin's items, and has a "nothing to sort" reject class. **Not trained or validated on medical waste**: check it on your own photos before trusting it. Records are tagged `zeroshot`. Force with `$env:MEDSORT_ENGINE="clip"`; disable with `$env:MEDSORT_ENGINE="off"`.
3. **Manual selection** if neither is available.

Live "Auto-detect": camera frames go to `/api/detect` (records nothing) every 1.5 s. When the same class is confidently seen on 3 consecutive frames, the item is recorded and sorted once; the next item is only detected after the previous one leaves the view (a frame below the threshold re-arms it).

## Troubleshooting
- **"No AI engine is loaded"**: run `setup_ai.bat` (installs torch + transformers, ~1 GB) and restart `python app.py`. The yellow box on the page and the start-up message in PowerShell show the exact reason. Until then, choose the bin manually.
- **"Camera permission is blocked"**: click the camera/lock icon in the address bar > Camera > Allow, then reload; Windows: Settings > Privacy & security > Camera. Open the site as `http://127.0.0.1:5000`, never as a file or network IP. If the camera still fails, use "Take photo (phone camera)" or upload an image.
- Camera busy: close Zoom/Teams or other tabs using it.
