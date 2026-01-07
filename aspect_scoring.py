import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline
INPUT_CSV = r"C:\Users\Naveen Manoharan\netflix-cinephile-metric\data\raw\processed\aspect_sentences.csv"
OUTPUT_CSV = r"C:\Users\Naveen Manoharan\netflix-cinephile-metric\data\raw\processed\aspect_scores.csv"


BATCH_SIZE = 32 
SENTIMENT_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0
}

DEVICE = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if DEVICE == 0 else "CPU")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=DEVICE
)

def normalize_0_100(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series([50] * len(series))
    return 100 * (series - series.min()) / (series.max() - series.min())

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} aspect sentences")

    sentiment_scores = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Scoring sentiment"):
        batch = df.iloc[i:i + BATCH_SIZE]

        results = sentiment_pipeline(batch["sentence"].tolist())

        for (_, row), result in zip(batch.iterrows(), results):
            label = result["label"].lower()
            score = SENTIMENT_MAP[label]

            sentiment_scores.append({
                "title": row["title"],
                "aspect": row["aspect"],
                "sentiment": score
            })

    sentiment_df = pd.DataFrame(sentiment_scores)

    agg_df = (
        sentiment_df
        .groupby(["title", "aspect"])["sentiment"]
        .mean()
        .reset_index()
    )

    agg_df["score"] = normalize_0_100(agg_df["sentiment"])

    final_df = agg_df[["title", "aspect", "score"]]
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\nSENTIMENT SCORING COMPLETE")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Title–aspect scores: {len(final_df)}")

if __name__ == "__main__":
    main()
