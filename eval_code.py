#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
import re

def extract_code_blocks(text):
    """
    Return ALL code blocks found in a text.
    """
    if not text:
        return []

    matches = re.findall(
        r"```(?:python|py)?\n(.*?)```",
        text,
        re.DOTALL,
    )
    # if matches: 
        # print(matches)
    return [m.strip() for m in matches if m.strip()]

def extract_last_code_from_all_steps(step_list):
    last_code = None

    for step in step_list:
        texts = []

        if step.get("small_model_step"):
            texts.append(step["small_model_step"])

        if step.get("base_model_step"):
            texts.append(step["base_model_step"])

        for text in texts:
            codes = extract_code_blocks(text)
            if codes:
                last_code = codes[-1]

    return [last_code] if last_code is not None else [""]

import ast
import pickle
def load_step_list(path):
    path = Path(path)
    if path.suffix == ".pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        return ast.literal_eval(raw)
    
def load_predictions_from_directory(input_dir: Path):
    benchmark_all = load_dataset(
        'json',
        data_files="../data/test6.jsonl",
        split="train",
    )
    benchmark_all = list(benchmark_all)

    subdirs = sorted(
        [p for p in input_dir.iterdir() if p.is_dir()],
        key=lambda p: int(p.name)
    )

    results = []

    for subdir, bench in zip(subdirs, benchmark_all):
        step_list = load_step_list(subdir / f"0.pickle")

        code_list = extract_last_code_from_all_steps(step_list)
        print(subdir, code_list)
        if not code_list:
            code_list = [""]

        results.append({
            "question_id": str(bench["question_id"]),
            "code_list": code_list,
        })

    return results


def load_test6_benchmark():
    dataset = load_dataset(
        'json',
        data_files="../data/test6.jsonl",
        split="train",
    )

    benchmark = []

    import json
    import base64
    import zlib

    for sample in dataset:
        public_cases = json.loads(sample["public_test_cases"])

        import pickle

        private_cases = pickle.loads(
            zlib.decompress(
                base64.b64decode(sample["private_test_cases"])
            )
        )

        if isinstance(private_cases, str):
            private_cases = json.loads(private_cases)
        all_cases = public_cases + private_cases

        inputs = [x["input"] for x in all_cases]
        outputs = [x["output"] for x in all_cases]

        benchmark.append({
            "question_id": str(sample["question_id"]),
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs,
            }),
        })

    return benchmark

# def evaluate_predictions(predictions):
#     benchmark_all = load_test6_benchmark()
#     pred_map = {str(x["question_id"]): x for x in predictions}

#     benchmark = [x for x in benchmark_all if str(x["question_id"]) in pred_map]
#     ordered_preds = [pred_map[str(x["question_id"])] for x in benchmark]

#     print(f"Loaded {len(benchmark_all)} problems from test6")
#     print(f"Matched {len(benchmark)} predictions")

#     assert len(benchmark) == len(predictions)

#     results = codegen_metrics(
#         samples=ordered_preds,
#         benchmark=benchmark,
#     )
#     return results

def extract_answer(result, options):
    step_str = result[-1]['step_str']
    if options == "lcb":
        # LCB returns code blocks; extract the first code block if possible.
        try:
            s = re.findall(r'```(?:python)?\n(.*?)```', step_str, re.DOTALL | re.IGNORECASE)[0]
        except Exception as ex:
            print(f"Exception: {ex}. Failed to extract codeblock:\n{step_str}")
            s = step_str
    elif options == "aime" or options == "math" or options == "gpqa":
        s = step_str
    else:
        raise NotImplementedError
    return s

def evaluate_predictions(predictions):
    benchmark_all = load_test6_benchmark()
    pred_map = {str(x["question_id"]): x for x in predictions}

    benchmark = [x for x in benchmark_all if str(x["question_id"]) in pred_map]
    ordered_preds = [pred_map[str(x["question_id"])] for x in benchmark]

    print(f"Loaded {len(benchmark_all)} problems from test6")
    print(f"Matched {len(benchmark)} predictions")

    assert len(benchmark) == len(predictions)

    # ✅ FIX HERE
    generations_list = [pred["code_list"] for pred in ordered_preds]

    print(type(generations_list[0]))  # should be list
    print(type(generations_list[0][0]))  # should be str

    results = codegen_metrics(benchmark, generations_list)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("lcb_eval.json"),
        help="Where to save converted LCB-format predictions.",
    )
    args = parser.parse_args()

    print("Loading predictions...")
    predictions = load_predictions_from_directory(args.input_dir)
    print(f"Total problems: {len(predictions)}")

    args.output_file.write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    print(f"Saved converted predictions to {args.output_file}")

    print("Running evaluation on test6...")
    results = evaluate_predictions(predictions)

    print("\nEvaluation results:")
    print(results)


if __name__ == "__main__":
    main()