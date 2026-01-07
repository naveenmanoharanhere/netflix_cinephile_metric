import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

INPUT_CSV = r"whatever man"
OUTPUT_CSV = r"or woman"

ASPECTS = [
    "direction",
    "cinematography",
    "screenplay",
    "acting",
    "editing"
]

BATCH_SIZE = 8         
CONFIDENCE_THRESHOLD = 0.40

DEVICE = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if DEVICE == 0 else "CPU")
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    device=DEVICE
)

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} sentences")

    records = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Aspect classification"):
        batch = df.iloc[i:i + BATCH_SIZE]

        results = classifier(
            batch["sentence"].tolist(),
            ASPECTS,
            multi_label=False
        )

        for (_, row), result in zip(batch.iterrows(), results):
            label = result["labels"][0]
            score = result["scores"][0]

            if score >= CONFIDENCE_THRESHOLD:
                records.append({
                    "title": row["title"],
                    "aspect": label,
                    "confidence": round(score, 3),
                    "sentence": row["sentence"]
                })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\nASPECT EXTRACTION COMPLETE")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Aspect-tagged sentences: {len(out_df)}")


if __name__ == "__main__":
    main()
    

