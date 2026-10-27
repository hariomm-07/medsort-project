"""Train MobileNetV2 on labelled medical-waste images.
Folders: dataset/{train,val,test}/{yellow,red,white,blue}/*.jpg   (test is optional but recommended)
Run:  python train_model.py --epochs 15
"""
import argparse, json, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from waste_config import CLASSES

class Ordered(datasets.ImageFolder):            # force the project class order (ImageFolder is alphabetical by default)
    def find_classes(self, d): return list(CLASSES), {c: i for i, c in enumerate(CLASSES)}

def evaluate(model, loader, dev):
    model.eval(); y, p = [], []
    with torch.no_grad():
        for x, t in loader: p += model(x.to(dev)).argmax(1).cpu().tolist(); y += t.tolist()
    return y, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset"); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (needs more data)")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tr_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1)), transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(.2, .2, .2), transforms.RandomRotation(15), transforms.ToTensor(), norm])
    ev_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    tr = Ordered(os.path.join(a.data, "train"), tr_tf); va = Ordered(os.path.join(a.data, "val"), ev_tf)
    if len(tr) == 0 or len(va) == 0: raise SystemExit("No images found. See README step 'Prepare the dataset'.")
    for c in CLASSES:
        if tr.targets.count(CLASSES.index(c)) == 0: raise SystemExit(f"No training images for class '{c}'.")
    trl = DataLoader(tr, a.batch, shuffle=True); val = DataLoader(va, a.batch)
    try: m = models.mobilenet_v2(weights=None if a.no_pretrained else models.MobileNet_V2_Weights.DEFAULT)
    except Exception as e: print("Pretrained weights unavailable (", e, ") - training from scratch."); m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(CLASSES)); m.to(dev)
    opt = torch.optim.Adam(m.parameters(), a.lr); lossf = nn.CrossEntropyLoss(); best = -1
    for ep in range(a.epochs):
        m.train()
        for x, t in trl:
            opt.zero_grad(); lossf(m(x.to(dev)), t.to(dev)).backward(); opt.step()
        y, p = evaluate(m, val, dev); acc = accuracy_score(y, p); print(f"epoch {ep+1}/{a.epochs}  val acc {acc:.3f}")
        if acc > best: best = acc; torch.save(m.state_dict(), "waste_model.pt")      # keep best weights by validation accuracy
    m.load_state_dict(torch.load("waste_model.pt", map_location=dev))
    tdir = os.path.join(a.data, "test"); held = Ordered(tdir, ev_tf) if os.path.isdir(tdir) and len(Ordered(tdir, ev_tf)) else None
    name = "test" if held else "val (no test folder: figures are optimistic because val chose the best epoch)"
    y, p = evaluate(m, DataLoader(held or va, a.batch), dev)
    rep = classification_report(y, p, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=range(len(CLASSES))).tolist()
    print(f"\nEvaluated on: {name}\n", classification_report(y, p, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred, order", CLASSES, ")\n", cm)
    json.dump({"evaluated_on": name, "accuracy": accuracy_score(y, p), "report": rep, "confusion_matrix": cm, "classes": CLASSES},
              open("metrics.json", "w"), indent=2)
    print("Saved waste_model.pt and metrics.json")

if __name__ == "__main__": main()
