import os
import pandas as pd
import re
import nltk
from tqdm import tqdm

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
INPUT_CSV = r"C:\Users\Naveen Manoharan\netflix-cinephile-metric\data\raw\imdb_reviews.csv"
OUTPUT_CSV = r"C:\Users\Naveen Manoharan\netflix-cinephile-metric\data\raw\processed\imdb_sentences.csv"
MIN_WORDS = 8
MAX_WORDS = 40
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)
def clean_sentence(sentence: str) -> str:
    sentence = EMOJI_PATTERN.sub("", sentence)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"[^A-Za-z0-9.,!?'\"]+", " ", sentence)
    return sentence.strip()
def is_valid_sentence(sentence: str) -> bool:
    words = sentence.split()

    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False

    if sentence.lower().startswith((
        "this review",
        "spoilers",
        "note:",
        "warning"
    )):
        return False

    return True

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = pd.read_csv(INPUT_CSV, encoding="latin1")

    print("Columns detected:", list(df.columns))

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        title = row["title"]
        review_text = str(row["content"])  

        sentences = sent_tokenize(review_text)

        for sentence in sentences:
            sentence = clean_sentence(sentence)

            if is_valid_sentence(sentence):
                records.append({
                    "title": title,
                    "sentence": sentence
                })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\nPREPROCESSING COMPLETE")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Total valid sentences: {len(out_df)}")

if __name__ == "__main__":
    main()