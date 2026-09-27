# Cattle Breed Classifier

A Flask web app around a pretrained MobileNetV2 transfer-learning model
(15-class softmax, 224x224x3 input) for classifying cattle breeds from
images, with confidence-based out-of-distribution (OOD) flagging.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## BEFORE you run this — 2 things to fix

1. **Class names** (`app.py`, `CLASS_NAMES` list): currently placeholders
   (`breed_0` ... `breed_14`). Replace them with your real breed names, in
   the EXACT order used during training (check your training notebook's
   `class_indices` or the folder order your data loader used). Wrong order
   = confidently wrong labels.

2. **OOD threshold** (`app.py`, `OOD_THRESHOLD = 0.50`): this decides when
   the app says "not confident enough" instead of showing a breed. Tune
   this against your own validation set — test with a genuinely
   out-of-distribution image (a dog, a car, a blurry photo) and see what
   confidence score it gets, then set the threshold just above that.

## How it works

1. `/` — upload form (GET)
2. `/predict` — receives the uploaded image (POST), preprocesses it to
   224x224 with MobileNetV2's `preprocess_input`, runs `model.predict()`,
   and returns the top prediction + top-3 breakdown.
3. If the top softmax probability is below `OOD_THRESHOLD`, the app shows
   an "uncertain" result instead of forcing a label — this is the
   OOD-awareness your resume already describes.

## Deploying (e.g. to Render, like your other projects)

- Add a `render.yaml` or set the start command to:
  `gunicorn app:app`
- Make sure `model/cattle_breed_model.keras` is committed to the repo (or
  use Git LFS — it's ~29MB, under GitHub's 100MB hard limit but worth
  checking your host's slug size limits).
- Set `debug=False` implicitly by using gunicorn instead of `app.run()`
  in production.

## Project structure

```
cattle-classifier/
├── app.py                          # Flask app + inference logic
├── model/
│   └── cattle_breed_model.keras    # your pretrained model
├── templates/
│   └── index.html                  # upload form + results
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```
