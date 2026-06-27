"""
SADIE GUI (Streamlit)

User-friendly front-end for:
- selecting / uploading Blue & Red policy docs
- running ingestion (policy_docs -> parameterized_output)
- training self-play PPO (v8)
- evaluating and viewing commander/DSS metrics

Run (from repo parent, recommended):
  streamlit run sadie_code/gui_app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_PARENT_DIR = PACKAGE_DIR.parent  # run `python -m sadie_code.*` from here

POLICY_DOCS_DIR = PACKAGE_DIR / "policy_docs"
BLUE_DOCS_DIR = POLICY_DOCS_DIR / "blue"
RED_DOCS_DIR = POLICY_DOCS_DIR / "red"

MODELS_DIR = PACKAGE_DIR / "models"


@dataclass(frozen=True)
class CmdResult:
    exit_code: int
    output: str


def _ensure_dirs() -> None:
    BLUE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _list_files(dir_path: Path, exts: tuple[str, ...]) -> list[Path]:
    if not dir_path.exists():
        return []
    out: list[Path] = []
    for p in sorted(dir_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def _save_uploads(uploaded: Iterable, target_dir: Path) -> list[Path]:
    saved: list[Path] = []
    for uf in uploaded:
        # `uf.name` comes from the browser; keep basename only
        safe_name = Path(uf.name).name
        out_path = target_dir / safe_name
        out_path.write_bytes(uf.getbuffer())
        saved.append(out_path)
    return saved


def _run_cmd(cmd: list[str], cwd: Path, env: Optional[dict[str, str]] = None) -> CmdResult:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return CmdResult(exit_code=int(proc.returncode), output=proc.stdout or "")


def _stream_cmd(cmd: list[str], cwd: Path) -> CmdResult:
    """
    Run a command and stream output into the UI.
    Blocks until completion (simple and robust for Streamlit).
    """
    placeholder = st.empty()
    buf: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line.rstrip("\n"))
        # Keep UI snappy: only render last ~400 lines
        tail = "\n".join(buf[-400:])
        placeholder.code(tail)
    rc = proc.wait()
    out = "\n".join(buf)
    return CmdResult(exit_code=int(rc), output=out)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(label: str, value: object) -> None:
    c1, c2 = st.columns([2, 3])
    c1.write(f"**{label}**")
    c2.write(value)


def main() -> None:
    st.set_page_config(page_title="SADIE GUI", layout="wide")
    _ensure_dirs()

    st.title("SADIE — Blue/Red Strategic Simulation (GUI)")
    st.caption(
        "Pick policy documents, run ingestion, train self-play PPO, and evaluate commander/DSS metrics. "
        "This GUI runs the same `python -m sadie_code.*` commands as the CLI."
    )

    with st.sidebar:
        st.subheader("Runtime")
        st.write(f"**Repo root:** `{REPO_PARENT_DIR}`")
        st.write(f"**Python:** `{sys.executable}`")
        st.write(f"**CWD:** `{Path.cwd()}`")
        st.divider()
        st.subheader("Model directory")
        model_dir = st.text_input(
            "Model output folder",
            value=str(MODELS_DIR / "v8_contest"),
            help="Training outputs and eval reports go here. Model .zip files are local-only (gitignored).",
        )
        st.session_state["model_dir"] = model_dir

    tab_policies, tab_train, tab_eval = st.tabs(["1) Policies", "2) Train", "3) Evaluate"])

    with tab_policies:
        st.subheader("Policy documents")
        st.write("Upload PDFs or place them under `sadie_code/policy_docs/blue/` and `sadie_code/policy_docs/red/`.")

        col_b, col_r = st.columns(2)
        with col_b:
            st.markdown("### Blue (US/NATO)")
            blue_upload = st.file_uploader(
                "Upload Blue docs (PDF/TXT/MD/DOCX)",
                type=["pdf", "txt", "md", "docx"],
                accept_multiple_files=True,
                key="blue_upload",
            )
            if st.button("Save Blue uploads", use_container_width=True, disabled=not blue_upload):
                saved = _save_uploads(blue_upload, BLUE_DOCS_DIR)
                st.success(f"Saved {len(saved)} file(s) to `{BLUE_DOCS_DIR}`.")
        with col_r:
            st.markdown("### Red (Adversary)")
            red_upload = st.file_uploader(
                "Upload Red docs (PDF/TXT/MD/DOCX)",
                type=["pdf", "txt", "md", "docx"],
                accept_multiple_files=True,
                key="red_upload",
            )
            if st.button("Save Red uploads", use_container_width=True, disabled=not red_upload):
                saved = _save_uploads(red_upload, RED_DOCS_DIR)
                st.success(f"Saved {len(saved)} file(s) to `{RED_DOCS_DIR}`.")

        st.divider()
        col_l, col_m, col_r2 = st.columns([2, 1, 2])
        with col_l:
            st.markdown("### Current Blue docs")
            blue_files = _list_files(BLUE_DOCS_DIR, (".pdf", ".txt", ".md", ".docx"))
            st.write("\n".join([f"- `{p.name}`" for p in blue_files]) if blue_files else "_None_")
        with col_r2:
            st.markdown("### Current Red docs")
            red_files = _list_files(RED_DOCS_DIR, (".pdf", ".txt", ".md", ".docx"))
            st.write("\n".join([f"- `{p.name}`" for p in red_files]) if red_files else "_None_")

        st.divider()
        st.subheader("Ingest policy docs → JSON parameters")
        st.write("Requires Ollama + LlamaIndex configured in `ingest_policy_docs.py`.")
        ingest_side = st.selectbox("Which side to ingest?", ["both", "blue", "red"])
        if st.button("Run ingestion", type="primary"):
            args = []
            if ingest_side in {"blue", "red"}:
                args = ["--side", ingest_side]
            cmd = [sys.executable, "-m", "sadie_code.ingest_policy_docs", *args]
            st.write("Running:")
            st.code(" ".join(cmd))
            res = _stream_cmd(cmd, cwd=REPO_PARENT_DIR)
            if res.exit_code == 0:
                st.success("Ingestion complete.")
            else:
                st.error(f"Ingestion failed (exit {res.exit_code}). See output above.")

    with tab_train:
        st.subheader("Train self-play PPO (v8)")
        st.write("This will take minutes to hours depending on iterations/steps.")

        col1, col2, col3 = st.columns(3)
        with col1:
            iterations = st.number_input("Iterations (rounds)", min_value=1, max_value=200, value=20, step=1)
        with col2:
            steps = st.number_input("Steps per round per side", min_value=5_000, max_value=2_000_000, value=60_000, step=5_000)
        with col3:
            eval_eps = st.number_input("Eval episodes per round", min_value=0, max_value=100, value=12, step=1)

        st.caption("Training eval is raw-policy (no COA diversity guard).")

        if st.button("Start training", type="primary"):
            Path(st.session_state["model_dir"]).mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "sadie_code.train_self_play_v8",
                "--model-dir",
                st.session_state["model_dir"],
                "--iterations",
                str(int(iterations)),
                "--steps",
                str(int(steps)),
                "--eval-episodes",
                str(int(eval_eps)),
            ]
            st.write("Running:")
            st.code(" ".join(cmd))
            res = _stream_cmd(cmd, cwd=REPO_PARENT_DIR)
            if res.exit_code == 0:
                st.success("Training complete (blue_final.zip / red_final.zip written).")
            else:
                st.error(f"Training failed (exit {res.exit_code}). See output above.")

    with tab_eval:
        st.subheader("Evaluate")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            games = st.number_input("Games", min_value=1, max_value=500, value=50, step=1)
        with col_b:
            diversity_guard = st.checkbox(
                "Enable eval diversity guard (DSS only)",
                value=False,
                help="Eval-only shim that breaks repeated identical COAs; not used for training validation.",
            )

        report_path = Path(st.session_state["model_dir"]) / "eval_report.json"
        if st.button("Run evaluation", type="primary"):
            cmd = [
                sys.executable,
                "-m",
                "sadie_code.evaluate",
                "--model-dir",
                st.session_state["model_dir"],
                "--games",
                str(int(games)),
            ]
            if diversity_guard:
                cmd.append("--diversity-guard")
            st.write("Running:")
            st.code(" ".join(cmd))
            res = _stream_cmd(cmd, cwd=REPO_PARENT_DIR)
            if res.exit_code == 0:
                st.success(f"Evaluation complete. Report: `{report_path}`")
            else:
                st.error(f"Evaluation failed (exit {res.exit_code}). See output above.")

        st.divider()
        st.subheader("Latest report")
        if report_path.exists():
            report = _load_json(report_path)
            cm = report.get("commander_metrics", {}) or {}

            st.markdown("### Commander / DSS metrics")
            _metric_row("Score margin mean (Blue − Red)", f"{cm.get('mean_score_margin', 0):+.3f}")
            _metric_row("Score margin std", f"{cm.get('std_score_margin', 0):.3f}")
            _metric_row("Tension-in-band rate (0.45–0.72)", f"{cm.get('tension_in_band_rate', 0)*100:.1f}%")
            _metric_row("Blue COA collapse rate (>80%)", f"{cm.get('blue_collapse_rate', 0)*100:.1f}%")
            _metric_row("Red COA collapse rate (>80%)", f"{cm.get('red_collapse_rate', 0)*100:.1f}%")
            _metric_row("Blue unique actions (mean)", f"{cm.get('mean_blue_unique_actions', 0):.1f}")
            _metric_row("Red unique actions (mean)", f"{cm.get('mean_red_unique_actions', 0):.1f}")
            _metric_row("Blue entropy (norm)", f"{cm.get('mean_blue_entropy_norm', 0):.2f}")
            _metric_row("Red entropy (norm)", f"{cm.get('mean_red_entropy_norm', 0):.2f}")

            st.markdown("### Outcome rates")
            _metric_row("Blue win rate", f"{report.get('blue_win_rate', 0)*100:.1f}%")
            _metric_row("Red win rate", f"{report.get('red_win_rate', 0)*100:.1f}%")
            _metric_row("Draw rate", f"{report.get('draw_rate', 0)*100:.1f}%")
            _metric_row("Mutual loss rate", f"{report.get('mutual_loss_rate', 0)*100:.1f}%")

            st.markdown("### Scores / rewards")
            _metric_row("Blue strategic score (final avg)", f"{report.get('mean_blue_strategic_score', 0):.3f}")
            _metric_row("Red strategic score (final avg)", f"{report.get('mean_red_strategic_score', 0):.3f}")
            _metric_row("Blue RL reward (episode avg)", f"{report.get('mean_blue_reward', 0):.2f}")
            _metric_row("Red RL reward (episode avg)", f"{report.get('mean_red_reward', 0):.2f}")

            st.markdown("### COA mix (top actions)")
            blue_hist = report.get("blue_action_hist") or []
            red_hist = report.get("red_action_hist") or []
            action_names = report.get("action_names")  # may not exist in older reports

            # If action_names missing, fall back to importing list at runtime.
            if not action_names:
                try:
                    from .eval_utils import ACTION_NAMES as _ACTIONS  # type: ignore

                    action_names = list(_ACTIONS)
                except Exception:
                    action_names = [f"action_{i}" for i in range(max(len(blue_hist), len(red_hist)))]

            def _top_k(hist: list[int], k: int = 8) -> list[tuple[str, int]]:
                pairs = list(enumerate(hist))
                pairs.sort(key=lambda x: x[1], reverse=True)
                out2: list[tuple[str, int]] = []
                for i, cnt in pairs[:k]:
                    if cnt > 0 and i < len(action_names):
                        out2.append((str(action_names[i]), int(cnt)))
                return out2

            colx, coly = st.columns(2)
            with colx:
                st.write("**Blue**")
                st.json({k: v for k, v in _top_k(blue_hist)})
            with coly:
                st.write("**Red**")
                st.json({k: v for k, v in _top_k(red_hist)})

            with st.expander("Raw report JSON"):
                st.json(report)
        else:
            st.info(f"No report found at `{report_path}`. Run evaluation first.")

        pub_dir = Path(st.session_state["model_dir"]) / "publication"
        st.divider()
        st.subheader("Publication exports (LaTeX tables + figures)")
        st.caption(
            "Written automatically on every evaluation to `<model-dir>/publication/`. "
            "CSV columns map to `tab:main_results`, `tab:coa_mix`, and `tab:diversity`."
        )
        pub_files = [
            ("Main results (CSV)", pub_dir / "table_main_results.csv"),
            ("COA mix (CSV)", pub_dir / "table_coa_mix.csv"),
            ("Diversity (CSV)", pub_dir / "table_diversity.csv"),
            ("Per-episode (CSV)", pub_dir / "episodes.csv"),
            ("All tables (JSON)", pub_dir / "publication_results.json"),
        ]
        for label, fp in pub_files:
            if fp.exists():
                st.download_button(
                    label=f"Download {label}",
                    data=fp.read_bytes(),
                    file_name=fp.name,
                    mime="text/csv" if fp.suffix == ".csv" else "application/json",
                    key=f"dl_{fp.name}",
                )
        fig_dir = pub_dir / "figures"
        if fig_dir.exists():
            figs = sorted(fig_dir.glob("*.png"))
            if figs:
                st.markdown("**Figures (PNG)**")
                cols = st.columns(min(2, len(figs)))
                for i, fig_path in enumerate(figs):
                    cols[i % len(cols)].image(str(fig_path), caption=fig_path.stem, use_container_width=True)


if __name__ == "__main__":
    main()

