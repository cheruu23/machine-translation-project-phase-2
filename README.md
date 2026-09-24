# English → Amharic Neural Machine Translation

Seq2Seq machine translation for the English–Amharic pair, comparing a basic
LSTM encoder–decoder against an attention-based one, with an error analysis,
attention visualisations, and a deployed translation API.

---

## 1. Repository layout

```          
├── deployment
│   ├── app.py               # Flask REST API + browser demo page
│   ├── nmt_translator.py    # architectures + greedy decoding + Translator
│   ├── requirements.txt
│   ├── test_api.py          # API smoke test
│   └── artifacts/
│       ├── attention_seq2seq_best.pt
│       ├── basic_seq2seq_best.pt
│       ├── en_tokenizer.model
│       └── am_tokenizer.model        
```

`data/` and `checkpoints_final/` are not committed (size). See §3.

---

## 2. Requirements

* Python 3.10+
* A CUDA GPU is optional. Training was done on an NVIDIA GeForce MX450 (2 GB VRAM);
  inference runs fine on CPU (~40 ms per sentence).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r deployment/requirements.txt
```

For the notebooks, additionally:

```bash
pip install jupyter torch sentencepiece sacrebleu matplotlib pandas numpy
```

Amharic text in the attention heatmaps needs an Ethiopic font:

| OS | Command |
|---|---|
| Ubuntu / Colab | `sudo apt-get install -y fonts-noto-core` |
| macOS | `brew install --cask font-noto-sans-ethiopic` |
| Windows | The bundled **Nyala** font is sufficient |

Without it, matplotlib prints `Glyph … missing from font(s) DejaVu Sans`
and every Amharic label is drawn as an empty box.

---

## 3. Data

Place the three CSV files in `data/`. Each must have the columns
`English` and `Amharic`:

```
data/train.csv
data/val.csv
data/test.csv
```

Preprocessing applied by the notebooks:

* English lowercased, whitespace collapsed
* Amharic NFC-normalised, whitespace collapsed, spacing normalised
  around the Ethiopic full stop (`።`, U+1362)
* rows with a null on either side dropped
* pairs with a source/target word-count ratio above 3:1 dropped
* exact duplicate pairs dropped
* exact `(English, Amharic)` pairs shared with validation/test removed from training
* tokenisation: SentencePiece **unigram**, 7 000 pieces per language,
  `pad=0, unk=1, bos=2, eos=3`

---

## 4. Running the notebooks

Run in order. Notebooks 2 and 3 only need the checkpoints from notebook 1.

### 4.1 Training — `NMT_Final_English_Amharic_Safe.ipynb`

Set `DATA_DIR` in cell 3, then run all. Trains both models and writes
`checkpoints_final/`. On a 2 GB GPU expect the attention model to take
substantially longer than the basic one.

### 4.2 First-pass evaluation — `NMT_Evaluation_Visualization_Only.ipynb`

Set `PROJECT_DIR`. Loads the saved checkpoints and reports BLEU, chrF,
test loss, inference time and parameter counts. Does not train.

### 4.3 Completion — `NMT_Part2_Completion.ipynb`

Set `PROJECT_DIR` in section 0. This notebook:

* installs an Ethiopic font and fixes the unreadable heatmaps
* audits the dataset and measures **train/test near-duplicate leakage**
* **retrains the basic baseline to the full epoch budget** so the comparison
  between the two models is fair (set `RETRAIN_BASIC = False` to skip)
* recomputes test loss correctly (teacher-forced *and* free-running) plus perplexity
* computes BLEU / chrF / chrF++ on the full test set and on the leak-free subset
* tags every test sentence across the eight required error categories
* produces three attention heatmaps with a numeric alignment readout
* builds and smoke-tests the deployment folder

All figures and tables land in `report_assets/`.

---

## 5. Running the deployed application

### 5.1 Flask REST API

```bash
cd deployment
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:5000
```

Point it at a different artifacts folder with an environment variable:

```bash
ARTIFACT_DIR=/path/to/artifacts PORT=8000 python app.py
```

**Endpoints**

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | browser demo page |
| `GET` | `/health` | model/device status |
| `POST` | `/translate` | translate one sentence |
| `POST` | `/translate_batch` | translate up to 128 sentences |

**Example**

```bash
curl -X POST http://127.0.0.1:5000/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "I am going to the university."}'
```

```json
{
  "text": "I am going to the university.",
  "translation": "ወደ ዩኒቨርሲቲ እሄዳለሁ።",
  "model": "attention",
  "latency_ms": 38.4
}
```

Pass `"model": "basic"` to query the baseline instead. Errors return HTTP 400
with an `error` field (empty input, unknown model) or HTTP 500 (inference failure).

Smoke test with the server running:

```bash
python test_api.py                            # or: python test_api.py http://host:port
```

### 5.2 Streamlit UI

```bash
cd deployment
streamlit run streamlit_app.py
```

Text box for the English input, the Amharic output, an optional side-by-side
baseline comparison, and an optional attention heatmap for the entered sentence.

### 5.3 Public URL for a live demo from Colab

Section 7 of the completion notebook can open an ngrok tunnel. Set
`USE_NGROK = True` and paste a free token from
`https://dashboard.ngrok.com`. The printed URL serves both the demo page
and the API.

---

## 6. Known limitations

* Trained on ~8.5k sentence pairs with a 96-dim embedding and 192 hidden units,
  constrained by 2 GB of VRAM. This is far below what English→Amharic needs;
  outputs are frequently fluent-looking but wrong.
* Greedy decoding only. No beam search, no length normalisation.
* Sources longer than 32 subword tokens are truncated.
* The mined corpus contains near-duplicate pairs across splits. The completion
  notebook quantifies this and reports leak-free scores alongside the headline ones.
* The basic baseline in the original run early-stopped at epoch 2; section 2
  of the completion notebook re-runs it for a valid comparison.

---

## 7. Attribution

Dataset source and licence: **fill in before submission** — the corpus origin,
its licence, and how the train/validation/test split was produced.
