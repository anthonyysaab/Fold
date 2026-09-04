"""Where the two interpreters live. One definition, two failure modes.

`tests/` and `tools/` both shell out to the CUDA training interpreter, and both
used to hard-code the path. That path has now moved twice, and the second move
was INVISIBLE: two torch-parity tests skipped forever because their
``skipUnless`` guard named a directory that no longer existed, on every
interpreter, silently (`tests/README.md`). One definition here means a third
move is one edit.

The two consumers need OPPOSITE behaviour when the interpreter is absent, and
both are right, so this module offers both and refuses to choose:

* :func:`cuda_python` returns the path whether or not it exists. A caller that
  wants to SKIP asks ``cuda_python().is_file()``. That is
  `tests/test_learned_policy_v8.py`: a machine without torch should not fail
  the suite.
* :func:`require_cuda_python` raises :class:`FileNotFoundError` naming the
  path. A tool that cannot do its job without torch should say so before it
  starts working, not partway through. That is `tools/dead_head_experiment.py`.

There is deliberately no third "return None when missing" form. That is the
shape that produced the silent skip: a falsy value flows into a guard, the
guard reads as "torch is unavailable here", and nobody learns the path was
simply wrong.

The directory was renamed from ``neural network training`` to ``training`` on
2026-09-04 when the nested `NN-Poker-training` repository was dissolved
(`.handoff/DECISIONS.md` section 6). It holds only the ``.venv`` now; the
trainer source is at `archive/nn-poker-training-2026-09-04/`. The path has no
spaces any more, but keep quoting it in shell recipes -- the repo root above it
still does.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The CUDA training interpreter: 3.11.9, torch 2.13.0+cu130, CUDA available.
#: Verified after the 2026-09-04 rename. `pyvenv.cfg` points at the system
#: interpreter rather than at itself, which is why the venv survived being
#: renamed; the ``Scripts/*.exe`` shims (``pip.exe``) did NOT, so always invoke
#: ``python.exe -m pip``, never the shim.
CUDA_PYTHON = REPO_ROOT / "training" / ".venv" / "Scripts" / "python.exe"

#: The stdlib interpreter the suite, the lint and the live path run on. 3.11.9,
#: deliberately WITHOUT torch -- `tests/test_runtime_layout.py` pins that
#: importing `engine` leaves torch out of ``sys.modules``.
STDLIB_PYTHON = Path(
    r"C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"
)


def cuda_python() -> Path:
    """The CUDA interpreter's path, present or not.

    Never raises. Callers that should degrade to a skip test
    ``cuda_python().is_file()`` themselves, so the absence is visible at the
    guard rather than hidden behind a sentinel.
    """
    return CUDA_PYTHON


def require_cuda_python() -> Path:
    """The CUDA interpreter's path, or :class:`FileNotFoundError`.

    For tools that cannot do their job without torch. Call it at start-up --
    after argument parsing, before any work -- so the failure lands before the
    first subprocess rather than partway through a run.
    """
    if not CUDA_PYTHON.is_file():
        raise FileNotFoundError(
            f"the CUDA training interpreter is not at {CUDA_PYTHON}. It moved "
            "on 2026-09-03 (into this working tree) and again on 2026-09-04 "
            "(renamed to training/); tools/interpreters.py is the single "
            "definition. Check .handoff/CONTEXT.md section 1."
        )
    return CUDA_PYTHON


__all__ = [
    "CUDA_PYTHON",
    "REPO_ROOT",
    "STDLIB_PYTHON",
    "cuda_python",
    "require_cuda_python",
]
