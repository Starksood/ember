"""Run SUITE_V2 (12 subjects, 204 attack probes + bystander & receipt gates) on real Mem0.

Requires optional Mem0 + local HuggingFace / Qdrant packages:
    pip install "ember-custody[mem0]" qdrant-client sentence-transformers

Uses ``infer=False`` by default so ingestion does not depend on an external LLM key
while exercising real Mem0 ``add()``, ``search(top_k=..., threshold=...)``,
``delete()``, and baseline ``delete_all(user_id=...)`` over local Qdrant + dense embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from ember.audit import SUITE_V2, mem0_backend_spec, run_benchmark
from ember.cli import format_benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ember SUITE_V2 against a real Mem0 + Qdrant backend.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name configured in Mem0",
    )
    parser.add_argument("--embedding-dims", type=int, default=384, help="Vector dimensionality for Qdrant")
    parser.add_argument("--top-k", type=int, default=10, help="Mem0 search top_k for attack probes")
    parser.add_argument("--bystander-limit", type=int, default=5, help="Mem0 search top_k for bystander checks")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Mem0 search minimum similarity threshold (Mem0 OSS default is 0.1; use 0.0 for zero floor)",
    )
    parser.add_argument("--infer", action="store_true", help="Enable Mem0 LLM fact inference (requires LLM creds)")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise SystemExit(
            "mem0ai is not installed. Run: pip install 'ember-custody[mem0]' qdrant-client sentence-transformers"
        ) from exc

    os.environ.setdefault("MEM0_TELEMETRY", "false")
    llm_config: dict[str, object] = {"model": "gpt-4o-mini"}
    if os.environ.get("OPENAI_API_KEY"):
        llm_config["api_key"] = os.environ["OPENAI_API_KEY"]
    elif not args.infer:
        # Mem0 initializes LlmFactory on startup even when infer=False skips LLM calls.
        llm_config["api_key"] = f"offline-placeholder-{os.getpid()}"

    with tempfile.TemporaryDirectory(prefix="ember_mem0_bench_") as td:
        root = Path(td)

        def make_memory(scenario, role: str):
            run_dir = root / f"{scenario.subject}_{role}"
            run_dir.mkdir(parents=True, exist_ok=True)
            return Memory.from_config(
                {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "path": str(run_dir / "qdrant"),
                            "collection_name": f"ember_{scenario.subject.lower()}_{role}",
                            "embedding_model_dims": args.embedding_dims,
                        },
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {"model": args.embedding_model},
                    },
                    "llm": {"provider": "openai", "config": llm_config},
                    "history_db_path": str(run_dir / "history.db"),
                }
            )

        spec = mem0_backend_spec(
            make_memory,
            embedding_model=args.embedding_model,
            vector_store="qdrant (local)",
            retrieval_mode="vector_semantic_cosine",
            probe_limit=args.top_k,
            bystander_limit=args.bystander_limit,
            threshold=args.threshold,
            infer=args.infer,
            extra_search_settings={"embedding_dims": args.embedding_dims},
        )
        report = run_benchmark(SUITE_V2, backend=spec)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_benchmark_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
