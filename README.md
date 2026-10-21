# MoodLens

MoodLens is a local web demo that predicts which of six emotions a short text may
express: anger, fear, joy, love, sadness, or surprise. It turns the included
`train.txt` dataset and notebook experiment into a simple interface: enter text
and click **Analyze emotion** to see a prediction and confidence estimate.

## Run locally

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app trains its TF-IDF/logistic-regression pipeline from `train.txt` when it
starts and caches the model for the current app process. Keep the dataset beside
`app.py`.

## Notes

- Input is limited to 5,000 characters.
- The classifier does not write input or predictions to disk.
- Confidence values are model estimates, not calibrated emotional certainty.
- This is an educational text-classification demo, not a clinical or
  mental-health tool.
