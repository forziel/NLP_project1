import json, os, re
from typing import Literal
from datasets import load_dataset
from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr, ValidationError
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

SYSTEM_PROMPT = (
    'Extract JSON with keys: "category" (World, Sports, Business, Sci/Tech), '
    '"sentiment_score" (float between -1.0 and 1.0), and "entities" (list of strings). '
    'Output ONLY valid JSON. No markdown or extra text.'
)


class NewsExtraction(BaseModel):
    category: Literal["World", "Sports", "Business", "Sci/Tech"]
    sentiment_score: StrictFloat | StrictInt = Field(..., ge=-1.0, le=1.0)
    entities: list[StrictStr]


def load_ag_news_samples(n_samples: int = 100, gold_path: str = "gold_entities.json") -> list[dict]:
    ds = load_dataset("fancyzhx/ag_news", split="test")
    n = min(n_samples, len(ds))

    gold_map = {}
    if os.path.exists(gold_path):
        with open(gold_path, "r", encoding="utf-8") as f:
            gold_map = {int(item["id"]): item.get("gold_entities", []) for item in json.load(f)}

    return [
        {
            "id": idx,
            "text": ds[idx]["text"],
            "gold_category": LABELS[ds[idx]["label"]],
            "gold_entities": gold_map.get(idx, []),
        }
        for idx in range(n)
    ]


def try_raw_json(raw: str) -> dict | None:
    try:
        res = json.loads(raw)
        return res if isinstance(res, dict) else None
    except Exception:
        return None


def clean_and_parse_json(raw: str) -> dict | None:
    if not raw or not isinstance(raw, str):
        return None
    if (direct := try_raw_json(raw)) is not None:
        return direct

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    cleaned = m.group(1) if m else raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw and "}" in raw else ""
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return try_raw_json(cleaned)


def validate_schema(data: dict | None) -> NewsExtraction | None:
    if not data:
        return None
    try:
        return NewsExtraction(**data)
    except ValidationError:
        return None


def compute_entity_f1(gold_lists: list[list[str]], pred_lists: list[list[str]]) -> float:
    tp = fp = fn = 0
    for golds, preds in zip(gold_lists, pred_lists):
        g_set = {g.strip() for g in golds if g.strip()}
        p_set = {p.strip() for p in preds if p.strip()}
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    return round((2 * tp / (2 * tp + fp + fn)) * 100, 2) if (2 * tp + fp + fn) > 0 else 0.0


def evaluate_experiment(records: list[dict]) -> dict:
    total = len(records)
    raw_ok = clean_ok = schema_ok = 0
    y_true, y_pred, g_ents, p_ents = [], [], [], []

    for r in records:
        raw = r.get("raw_output", "")
        raw_ok += try_raw_json(raw) is not None
        cleaned = clean_and_parse_json(raw)
        clean_ok += cleaned is not None
        valid = validate_schema(cleaned)
        schema_ok += valid is not None

        y_true.append(r["gold_category"])
        y_pred.append(valid.category if valid else "INVALID")
        g_ents.append(r.get("gold_entities", []))
        p_ents.append(valid.entities if valid else [])

    return {
        "Raw JSON (%)": round(raw_ok / total * 100, 2) if total else 0.0,
        "Cleaned JSON (%)": round(clean_ok / total * 100, 2) if total else 0.0,
        "Schema Valid (%)": round(schema_ok / total * 100, 2) if total else 0.0,
        "Class Acc (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Class Macro-F1 (%)": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) * 100, 2),
        "Entity Exact-F1 (%)": compute_entity_f1(g_ents, p_ents),
    }