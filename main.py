import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer


HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "ChandlerLee/jeopardy-classifier")
CLASSIFIER_THRESHOLD = os.getenv("CLASSIFIER_THRESHOLD", "0.6289")
CLASSIFIER_THRESHOLD_SET = "CLASSIFIER_THRESHOLD" in os.environ
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_LENGTH = 256
MODEL_CACHE_DIR = Path("model_cache")

app = FastAPI()

model: DebertaV2ForSequenceClassification | None = None
tokenizer: DebertaV2Tokenizer | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bad_threshold = float(CLASSIFIER_THRESHOLD)
model_loaded = False


class Question(BaseModel):
    category: str
    question: str
    answer: str


class ValidationResult(BaseModel):
    valid: bool
    confidence: float
    reason: str | None = None


def format_question_text(item: Question) -> str:
    return f"Category: {item.category} | Question: {item.question} | Answer: {item.answer}"


def load_threshold_from_config(model_dir: Path) -> float | None:
    config_path = model_dir / "threshold_config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    threshold = config.get("bad_threshold")
    return float(threshold) if threshold is not None else None


@app.on_event("startup")
def startup_event() -> None:
    global bad_threshold, model, model_loaded, tokenizer

    if (MODEL_CACHE_DIR / "config.json").exists():
        print("Model cache found, skipping download.")
        model_dir = MODEL_CACHE_DIR
    else:
        model_dir = Path(
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                token=HF_TOKEN,
                local_dir=str(MODEL_CACHE_DIR),
                local_dir_use_symlinks=False,
            )
        )

    config_threshold = load_threshold_from_config(model_dir)
    if config_threshold is not None:
        bad_threshold = config_threshold
    if CLASSIFIER_THRESHOLD_SET:
        bad_threshold = float(CLASSIFIER_THRESHOLD)

    tokenizer = DebertaV2Tokenizer.from_pretrained(str(model_dir))
    model = DebertaV2ForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    warmup_inputs = tokenizer(
        [format_question_text(Question(category="History", question="Who was the first president of the United States?", answer="George Washington"))],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    warmup_inputs = {key: value.to(device) for key, value in warmup_inputs.items()}

    with torch.no_grad():
        _ = model(**warmup_inputs)

    model_loaded = True


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {
        "ok": True,
        "model_loaded": model_loaded,
        "device": device.type,
    }


@app.post("/validate", response_model=list[ValidationResult])
def validate(questions: list[Question]) -> list[ValidationResult]:
    if not model_loaded or model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not questions:
        return []

    texts = [format_question_text(item) for item in questions]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

    bad_probabilities = probabilities[:, 1].detach().cpu().tolist()
    results: list[ValidationResult] = []
    for bad_probability in bad_probabilities:
        is_valid = bad_probability < bad_threshold
        results.append(
            ValidationResult(
                valid=is_valid,
                confidence=round(1.0 - bad_probability if is_valid else bad_probability, 6),
                reason=None if is_valid else "Question failed classifier threshold",
            )
        )

    return results
