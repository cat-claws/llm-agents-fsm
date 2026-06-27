"""Helpers for resetting git-learning-lab checkouts from Git agents."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INSTALLER_NAME = "git-learning-installer.sh"
_BUNDLED_INSTALLER = Path(__file__).resolve().with_name(DEFAULT_INSTALLER_NAME)
RESET_TIMEOUT_SECONDS = 120
MAX_RESET_OUTPUT_CHARS = 8000


@dataclass(frozen=True)
class GitLearningResetResult:
    ok: bool
    message: str


def reset_git_learning_lab(
    work_dir: str | Path,
    *,
    installer: str | Path | None = None,
    timeout: int = RESET_TIMEOUT_SECONDS,
) -> GitLearningResetResult:
    """Run git-learning-installer.sh reset from the lab repo's parent."""

    work_path = Path(work_dir).resolve()
    repo_root, error = _git_toplevel(work_path)
    if error is not None:
        return GitLearningResetResult(False, error)

    assert repo_root is not None
    if repo_root.name != "git-learning-lab":
        return GitLearningResetResult(
            False,
            "[error] reset is only supported from a git-learning-lab checkout.\n"
            "Current git root: %s" % repo_root,
        )

    install_dir = repo_root.parent
    installer_path = _resolve_installer(install_dir, installer)
    if installer_path is None:
        return GitLearningResetResult(
            False,
            "[error] could not find %s.\n"
            "Checked bundled installer: %s\n"
            "Checked lab parent: %s\n"
            "Set GIT_LEARNING_INSTALLER=/path/to/%s to override."
            % (
                DEFAULT_INSTALLER_NAME,
                _BUNDLED_INSTALLER,
                install_dir / DEFAULT_INSTALLER_NAME,
                DEFAULT_INSTALLER_NAME,
            ),
        )

    try:
        proc = subprocess.run(
            ["bash", str(installer_path), "reset"],
            cwd=str(install_dir),
            input="y\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return GitLearningResetResult(
            False,
            "[error] git-learning reset timed out after %ss" % timeout,
        )
    except OSError as exc:
        return GitLearningResetResult(False, "[error] %s" % exc)

    if proc.returncode == 0:
        _restore_process_cwd(work_path, repo_root)

    output = _clip("\n".join(filter(None, [(proc.stdout or "").strip(), (proc.stderr or "").strip()])))
    if not output:
        output = "(no output)"
    message = "\n".join(
        [
            "Ran: bash %s reset" % installer_path,
            "From: %s" % install_dir,
            output,
            "[exit %s]" % proc.returncode,
        ]
    )
    return GitLearningResetResult(proc.returncode == 0, message)


def _git_toplevel(work_dir: Path) -> tuple[Path | None, str | None]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError as exc:
        return None, "[error] %s" % exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        suffix = "\n%s" % detail if detail else ""
        return None, "[error] current working directory is not inside a git repo: %s%s" % (work_dir, suffix)

    root_text = (proc.stdout or "").strip()
    if not root_text:
        return None, "[error] git did not report a repository root for %s" % work_dir
    return Path(root_text).resolve(), None


def _resolve_installer(install_dir: Path, installer: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if installer is not None:
        candidates.append(Path(installer))
    env_installer = os.environ.get("GIT_LEARNING_INSTALLER")
    if env_installer:
        candidates.append(Path(env_installer))
    candidates.append(_BUNDLED_INSTALLER)
    candidates.append(install_dir / DEFAULT_INSTALLER_NAME)

    for candidate in candidates:
        path = candidate if candidate.is_absolute() else install_dir / candidate
        if path.is_file():
            return path.resolve()
    return None


def _restore_process_cwd(work_dir: Path, repo_root: Path) -> None:
    for path in (work_dir, repo_root):
        try:
            if path.exists():
                os.chdir(str(path))
                return
        except OSError:
            continue


def _clip(text: str) -> str:
    if len(text) <= MAX_RESET_OUTPUT_CHARS:
        return text
    clipped = len(text) - MAX_RESET_OUTPUT_CHARS
    return text[:MAX_RESET_OUTPUT_CHARS] + "\n... [truncated: %s more chars]" % clipped
