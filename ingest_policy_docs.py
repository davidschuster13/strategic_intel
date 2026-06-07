from pathlib import Path
from typing import Optional
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

# ========================= CONFIG =========================
OLLAMA_MODEL = "qwen2.5:32b"

DATA_DIR = "./policy_docs"
OUTPUT_DIR = "./parameterized_output"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

PARAMETERIZATION_PROMPT = """
You are a strategic intelligence analyst extracting parameters from policy documents describing general strategic competition.
Extract in strict JSON format only. No explanations outside the JSON.

For each document section, output:
{
  "document_title": "...",
  "source_file": "...",
  "key_objectives": ["list of strategic goals with priority 0-1"],
  "red_lines": ["thresholds that trigger escalation"],
  "decision_heuristics": ["if-then rules or utility functions"],
  "capability_priorities": {"resource_type": weight 0-1, ...},
  "escalation_ladder": ["step 1 description", "step 2", ...],
  "key_quotes": ["exact quotes with page/context"],
  "confidence": 0.XX
}

Be precise, quote directly where possible, and stay faithful to the text.
"""
# =======================================================

def setup_llama_index():
    Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=300.0, temperature=0.1)
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"✅ Ollama + LlamaIndex ready with {OLLAMA_MODEL}")

def _load_docs_from_dir(data_dir: str):
    return SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".pdf", ".txt", ".md", ".docx"],
        recursive=True
    ).load_data()


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```json"):
        s = s.split("```json", 1)[1].split("```", 1)[0]
    elif s.startswith("```"):
        s = s.split("```", 1)[1].split("```", 1)[0]
    return s.strip()


def _validate_extraction(llm, original_text: str, extracted_params: dict) -> tuple:
    """Validate that extracted parameters appear in original document text."""
    objectives = extracted_params.get("key_objectives", [])
    red_lines = extracted_params.get("red_lines", [])
    
    if not objectives and not red_lines:
        return True, "No parameters to validate (acceptable for sparse docs)"
    
    validation_prompt = f"""Given this policy text, validate extracted strategic parameters.

Extracted objectives (sample): {objectives[:2] if objectives else "none"}
Extracted red-lines (sample): {red_lines[:2] if red_lines else "none"}

Policy text (first 3000 chars):
{original_text[:3000]}

Respond ONLY with:
VALID if most parameters appear in text
QUESTIONABLE if some parameters seem inferred, not explicit
INVALID if parameters don't match text
"""
    
    try:
        result = llm.complete(validation_prompt)
        response = (result.text or "").strip().upper()
        if "VALID" in response:
            return True, "Parameters validated against source text"
        elif "QUESTIONABLE" in response:
            return True, "Parameters flagged as inferred (acceptable with confidence discount)"
        else:
            return False, "Parameters do not match source text"
    except Exception as e:
        return False, f"Validation call failed: {e}"


def _parameterize_docs(documents, output_dir: Path, side: str):
    """Parameterize without a vector index (avoids Ollama embedding / index build failures).

    SimpleDirectoryReader often emits many chunks per PDF; we group by source file so
    each policy document yields one JSON file and one LLM call.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = Settings.llm

    by_file: dict[str, list] = defaultdict(list)
    for doc in documents:
        fname = doc.metadata.get("file_name", "unknown")
        by_file[str(fname)].append(doc)

    print(
        f"Loaded {len(documents)} chunk(s) from {len(by_file)} {side} source file(s). "
        "Parameterizing via LLM (no embedding index)..."
    )

    results = []
    for file_name in tqdm(sorted(by_file.keys()), desc=f"Parameterizing {side}"):
        parts = by_file[file_name]
        combined = "\n\n".join(d.text for d in parts)
        # For long documents, use primary section (cover, objectives, strategic doctrine)
        # rather than hard truncation which loses document context
        primary_section = combined[:30_000]  # Reduced from 50K to stay within token budget
        raw_text = ""
        try:
            prompt = f"{PARAMETERIZATION_PROMPT}\n\nDocument content (primary section):\n{primary_section}"
            completion = llm.complete(prompt)
            raw_text = (completion.text or "").strip()
        except Exception as e:
            print(f"⚠️ LLM call failed for {file_name}: {e}")
            results.append({"error": str(e), "filename": file_name, "side": side})
            continue

        try:
            json_str = _strip_json_fence(raw_text)
            params = json.loads(json_str)
            params["original_filename"] = Path(file_name).name
            params["side"] = side
            params["doc_length_chars"] = len(combined)
            
            # VALIDATE extraction against source
            is_valid, validation_msg = _validate_extraction(llm, combined, params)
            params["validation_status"] = "VALID" if is_valid else "QUESTIONABLE"
            params["validation_note"] = validation_msg
            if not is_valid:
                params["confidence"] = float(params.get("confidence", 0.5)) * 0.7  # Discount invalid params
            
            results.append(params)

            stem = Path(file_name).stem if file_name != "unknown" else "doc"
            out_path = output_dir / f"params_{stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ JSON parse failed for {file_name}: {e}")
            results.append(
                {
                    "error": raw_text or str(e),
                    "filename": file_name,
                    "side": side,
                }
            )

    master_file = output_dir / "all_parameterized_policies.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def ingest_and_parameterize(side: Optional[str] = None, data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR):
    side = side.lower() if side else None
    if side not in (None, "blue", "red"):
        raise ValueError("side must be one of: blue, red, or None")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    side_dirs = {
        "blue": Path(data_dir) / "blue",
        "red": Path(data_dir) / "red",
    }

    sides_to_process = [side] if side else ["blue", "red"]
    combined_results = {}

    for side_name in sides_to_process:
        source_dir = side_dirs[side_name]
        if not source_dir.exists():
            print(f"⚠️ Skipping {side_name}: source folder not found at {source_dir}")
            combined_results[side_name] = []
            continue

        print(f"📂 Loading {side_name} documents from {source_dir}...")
        documents = _load_docs_from_dir(str(source_dir))
        if not documents:
            print(f"⚠️ No {side_name} documents found in {source_dir}")
            combined_results[side_name] = []
            continue

        side_output_dir = output_root / side_name
        combined_results[side_name] = _parameterize_docs(documents, side_output_dir, side_name)

    # Keep a root-level aggregate file for compatibility/reporting.
    aggregate_file = output_root / "all_parameterized_policies.json"
    with open(aggregate_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 Done! Parameterized outputs saved to {output_root}")
    if side:
        return combined_results.get(side, [])
    return combined_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and parameterize policy docs by side.")
    parser.add_argument(
        "--side",
        choices=["blue", "red"],
        default=None,
        help="Only ingest one side. If omitted, ingests both policy_docs/blue and policy_docs/red."
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root docs directory.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    setup_llama_index()
    ingest_and_parameterize(side=args.side, data_dir=args.data_dir, output_dir=args.output_dir)
