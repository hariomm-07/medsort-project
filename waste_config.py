"""Single source of truth for class order. Used by train_model.py, app.py and (via /api/config) the frontend."""
CLASSES = ["yellow", "red", "white", "blue"]          # class_id 0..3
BIN_NAMES = {"yellow": "Yellow", "red": "Red", "white": "White translucent", "blue": "Blue"}
STATUS_SORTED = "Virtually Sorted"
STATUS_REVIEW = "Manual Verification Required"
