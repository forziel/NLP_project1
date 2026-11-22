import argparse, json, time
import ollama
from utils import NewsExtraction, SYSTEM_PROMPT, load_ag_news_samples

CUSTOM_EXAMPLES = [
    {"id": "c1", "text": "Apple launched the M4 MacBook Pro with faster AI neural engines.", "gold_category": "Sci/Tech", "gold_entities": ["Apple"]},
    {"id": "c2", "text": "Liverpool defeated Real Madrid 2-0 in the Champions League match at Anfield.", "gold_category": "Sports", "gold_entities": ["Liverpool", "Real Madrid", "Champions League", "Anfield"]},
    {"id": "c3", "text": "Federal Reserve kept interest rates unchanged amid steady inflation data.", "gold_category": "Business", "gold_entities": ["Federal Reserve"]},
    {"id": "c4", "text": "UN Security Council called an emergency meeting in New York over peace talks.", "gold_category": "World", "gold_entities": ["UN Security Council", "New York"]},
    {"id": "c5", "text": "Nvidia signed a $500M sponsorship deal with Manchester City to build a virtual stadium.", "gold_category": "Business", "gold_entities": ["Nvidia", "Manchester City"]},
]


def main(mode: str, n_samples: int, model: str, temp: float, custom: bool) -> None:
    samples = CUSTOM_EXAMPLES if custom else load_ag_news_samples(n_samples=n_samples)
    suffix = f"_t{temp}" if temp != 0.0 else ""
    suffix += "_custom" if custom else ""
    output_path = f"results_rmlower{mode}{suffix}.json"
    schema = NewsExtraction.model_json_schema() if mode == "schema" else None

    print(f"[{mode.upper()}] Model={model} | Temp={temp} | Samples={len(samples)}")
    results, start = [], time.perf_counter()

    for i, s in enumerate(samples, 1):
        t0 = time.perf_counter()
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"News Text:\n{s['text']}"},
                ],
                "options": {"temperature": temp, "num_ctx": 2048},
            }
            if schema is not None:
                kwargs["format"] = schema

            raw = ollama.chat(**kwargs)["message"]["content"]
        except Exception as e:
            print(f"[{i}/{len(samples)}] Error: {e}")
            raw = ""

        print(f"[{i}/{len(samples)}] {s['id']} ({time.perf_counter() - t0:.2f}s)")
        if custom:
            print(f"   Output: {raw.strip()}")
        results.append({**s, "raw_output": raw})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done ({time.perf_counter() - start:.2f}s) -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prompt", "schema"], required=True)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--model", default="qwen2.5:7b-instruct-q5_K_M")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--custom", action="store_true")
    args = p.parse_args()
    main(args.mode, args.n_samples, args.model, args.temp, args.custom)