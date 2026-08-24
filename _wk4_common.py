"""Shared helpers for the Wk4 linear-regression notebooks.

Plain module (not jupytext-paired, not matched by create_student_versions.py's
`wk4_*_soln.py` glob) so the load step doesn't need to be repeated as a
solution block in every notebook.

Every notebook imports this module directly (`from _wk4_common import
load_disney_df`), with no path-fixup code of its own. That works in two
setups without changing the import itself:

- **Google Colab:** a short bootstrap cell at the top of each notebook
  fetches this file from the public `MSAI-Business-Content-Public` GitHub
  repo into the Colab working directory before the `from _wk4_common
  import ...` line runs, so Python's default same-directory import
  resolution then finds it -- no student upload needed.
- **Local/VS Code:** `scripts/sync_jupytext.py` copies this file into
  `output/wk4/` on every sync, so it ends up sitting right next to the
  notebook there too, and the same same-directory import resolution just
  works.
"""

import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

DATASET_FILENAME = "disney_plus_synthetic.feather"

# Raw-file base URL for the public repo that mirrors runtime-fetchable
# content (dataset + helper modules) for Colab. Tracks `main`, not a pinned
# commit.
PUBLIC_REPO_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/blubin/MSAI-Business-Content-Public/main"
)


def _find_repo_root(start: Path) -> Path:
    return next(
        (p for p in [start, *start.parents] if (p / "requirements.txt").exists()),
        Path.cwd(),
    )


def _fetch_from_public_repo(filename: str, destination: Path) -> None:
    """Download `filename` from the public repo and save it to `destination`."""
    url = f"{PUBLIC_REPO_RAW_BASE_URL}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not fetch {filename} from {url}: {e}. Check your network "
            "connection, or upload the file manually into this notebook's "
            "working directory as a fallback."
        ) from e
    destination.write_bytes(data)


def load_disney_df() -> pl.DataFrame:
    """Load the synthetic Disney+ dataset used throughout the Wk4 notebooks.

    Looks for the dataset file in the current working directory first (the
    Colab setup, once a prior run has already fetched or the student has
    manually uploaded it there). Falls back to this repo's
    `output/disney_dataset/` directory (the local/VS Code setup, where the
    notebook runs from `output/wk4/` and the dataset lives in a sibling
    directory). If neither is found, fetches the dataset from the public
    `MSAI-Business-Content-Public` GitHub repo and caches it to the current
    working directory, so later calls in the same Colab session reuse the
    downloaded file instead of re-fetching it.
    """
    local_path = Path.cwd() / DATASET_FILENAME
    if local_path.exists():
        return pl.read_ipc(local_path)

    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    feather_path = repo_root / "output" / "disney_dataset" / DATASET_FILENAME
    if feather_path.exists():
        return pl.read_ipc(feather_path)

    _fetch_from_public_repo(DATASET_FILENAME, local_path)
    return pl.read_ipc(local_path)
