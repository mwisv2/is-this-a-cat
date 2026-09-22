# is this a cat

A small CNN that determines if an image has a cat, or no cat. Runs in your browser or from the command line.
The site is in `web/`. Python is for serving, classifying, and training.

## Run locally

```bash
git clone https://github.com/mwisv2/is-this-a-cat.git
cd is-this-a-cat
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Opens [http://127.0.0.1:8765/](http://127.0.0.1:8765/) and serves the `web/` folder. Weights in `web/weights.json`.

Classify an image from the command line:

```bash
python run.py path/to/photo.jpg
```

## Train it yourself

Needs numpy, h5py, and Pillow. Downloads the catvnoncat h5 sets into `.cache/` (and a few extra pet images), trains, writes `web/weights.json`.

```bash
source .venv/bin/activate
pip install -r requirements.txt
python run.py train
```

Then refresh the local site. Retrain whenever you change the net or data.

