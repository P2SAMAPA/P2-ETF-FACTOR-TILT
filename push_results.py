# push_results.py — P2-ETF-FACTOR-TILT
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from config import HF_RESULTS_REPO, HF_RESULTS_FILE

log = logging.getLogger(__name__)


def push_to_hf(df: pd.DataFrame, run_date: str) -> None:
    """
    Save results as parquet and push to HuggingFace dataset repo.
    Requires HF_TOKEN environment variable (set in GitHub Actions secrets).
    Appends to existing results if the file already exists, keeping
    the last 252 run_date entries (1 trading year).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set — skipping push")
        return

    api = HfApi(token=token)

    # ── Try to load existing results and append ───────────────────────────────
    try:
        from huggingface_hub import hf_hub_download
        existing_path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=HF_RESULTS_FILE,
            repo_type="dataset",
            token=token,
        )
        existing_df = pd.read_parquet(existing_path)
        log.info(f"Existing results loaded: {existing_df.shape}")

        # Remove today's date if already present (idempotent re-runs)
        existing_df = existing_df[existing_df["run_date"] != run_date]

        combined = pd.concat([existing_df, df], ignore_index=True)

        # Keep last 252 unique run_dates
        unique_dates = sorted(combined["run_date"].unique())
        if len(unique_dates) > 252:
            keep_dates = unique_dates[-252:]
            combined   = combined[combined["run_date"].isin(keep_dates)]

        log.info(f"Combined results: {combined.shape} ({combined['run_date'].nunique()} run dates)")

    except Exception as e:
        log.info(f"No existing results (first run): {e}")
        combined = df

    # ── Save and push ─────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / HF_RESULTS_FILE
        combined.to_parquet(local_path, index=False)
        log.info(f"Saved parquet: {local_path} ({local_path.stat().st_size / 1024:.1f} KB)")

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=HF_RESULTS_FILE,
            repo_id=HF_RESULTS_REPO,
            repo_type="dataset",
            commit_message=f"Daily factor tilt results — {run_date}",
        )
        log.info(f"Pushed to {HF_RESULTS_REPO}/{HF_RESULTS_FILE}")
