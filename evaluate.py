import argparse, json
from utils import evaluate_experiment


def main(prompt_path: str, schema_path: str, gold_path: str) -> None:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_records = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_records = json.load(f)

    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            gold_map = {int(item["id"]): item.get("gold_entities", []) for item in json.load(f)}
        for rec in prompt_records + schema_records:
            rec_id = int(rec["id"])
            if rec_id in gold_map:
                rec["gold_entities"] = gold_map[rec_id]
    except (FileNotFoundError, ValueError):
        pass

    res_prompt = evaluate_experiment(prompt_records)
    res_schema = evaluate_experiment(schema_records)

    print(f"\n{'Metric':<24} | {'Prompt-Only':>14} | {'Schema-Constrained':>18}")
    print("-" * 62)
    for metric in res_prompt:
        print(f"{metric:<24} | {res_prompt[metric]:>13.2f}% | {res_schema[metric]:>17.2f}%")
    print("-" * 62)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="results_prompt.json")
    p.add_argument("--schema", default="results_schema.json")
    p.add_argument("--gold", default="gold_entities.json")
    args = p.parse_args()
    main(args.prompt, args.schema, args.gold)