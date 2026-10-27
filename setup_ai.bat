@echo off
echo Installing MedSort with AI support (first run downloads torch and the CLIP model, about 1 GB)...
python -m pip install --upgrade pip
python -m pip install flask flask-cors pillow torch torchvision transformers scikit-learn
echo.
echo Done. Start the site with:  python app.py
pause
