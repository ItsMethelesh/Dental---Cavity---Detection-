# Dental Cavity Detection (X-ray)

This project builds a dental cavity detection workflow using:

- Python
- OpenCV (candidate region detection)
- CNN (PyTorch classifier)
- Streamlit (attractive UI)

It works with your existing `data/` folder of X-ray images and supports two training modes:

1. `labels.csv` provided (supervised labels)
2. no labels file (pseudo labels generated from OpenCV cavity-candidate scores)

## 1. Install

```bash
python -m pip install -r requirements.txt
```

If you see Windows long-path errors while installing `torch`, use a short-path virtual environment:

```powershell
python -m venv D:\vdc
D:\vdc\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Optional labels file

If you have real labels, create `labels.csv` in the project root:

```csv
image,label
i1.png,0
i2.png,1
```

Accepted label values: `0/1`, `healthy/cavity`, `false/true`, `no/yes`.

## 3. Train CNN

```bash
python train.py --data-dir data --output models/cavity_cnn.pt --epochs 14
```

Useful options:

- `--labels-csv labels.csv`
- `--max-samples 500`
- `--pseudo-quantile 0.62`

Training creates:

- model checkpoint: `models/cavity_cnn.pt`
- epoch history: `models/cavity_cnn.history.json`
- split records: `models/cavity_cnn.records.csv`

## 4. Run UI

```bash
streamlit run app.py
```

Frontend flow:

- Upload one dental X-ray image
- Click `Analyze Image`
- View risk score, highlighted regions, masks, and signal breakdown

Model training is backend-only (`train.py`) and is not exposed in the frontend UI.
If PyTorch is not installed yet, the app still runs in OpenCV-only fallback mode for analysis.

## 5. How highlighting works

- OpenCV generates candidate lesion regions from dark/texture anomalies.
- CNN predicts cavity probability.
- Grad-CAM localizes discriminative areas.
- Final overlay combines OpenCV candidates + Grad-CAM mask.

## Notes

- Some files in your dataset are exact duplicates; the training pipeline removes duplicates by hash.
- This is a decision-support prototype and not a medical diagnosis system.
