# Notes

Working notes for ReviewGraph. Decisions, problems and their fixes go here as they happen.

## AI usage log

| Date | Tool | Used for |
|------|------|----------|
| 2026-11-11 | Claude Code | Wrote `implementation-plan.html`, the exercise-by-exercise guide I follow for this project. |
| 2026-11-12 | Claude Code | Exercise 0.1: initialised the git repository, wrote `.gitignore` and started this file. |
| 2026-11-12 | Claude Code | Exercise 0.2: created the venv, found why the lab constraints were ignored on my machine, produced `requirements.txt`. |
| 2026-11-12 | Claude Code | Exercise 0.3: package skeleton, `pyproject.toml`, `.env.example`, filled in `get_settings()` in `config.py`, wrote `tests/test_config.py`. |
| 2026-11-12 | Claude Code | Exercise 0.4: wrote `scripts/get_target.py`, worked out which of the 12 commits in the range are the eight increment commits. |
| 2026-11-12 | Claude Code | Exercise 1.1: filled in `GitRepo` in `gitrepo.py`, wrote the tiny-repository fixture and `tests/test_gitrepo.py`. |
| 2026-11-12 | Claude Code | Exercise 1.2: `Unit`, `is_reviewable` and `build_units` in `schemas.py`, `scripts/build_units.py`, `tests/test_units.py`. |

## Decisions

- 2026-11-12: There is no template repository, so the repository was created with `git init` in this folder. Work happens directly on `main` instead of a `solution` branch.
- 2026-11-12: Python 3.12.7 (x64) in `.venv`. `requirements.txt` is a `pip freeze` of an install constrained by the lab 3 requirements file, so shared packages have the lab versions: langchain 1.4.2, langchain-core 1.6.5, langgraph 1.2.12, langchain-ollama 1.1.0, pydantic 2.13.5, google-genai 2.25.0. Packages the lab file does not contain: langchain-google-genai 4.4.0 (no conflict with langchain-core 1.6.5), python-dotenv 1.2.4, pytest 9.2.0.
- 2026-11-12: The range `29949a3..e594bc8` has 12 commits for 8 increments. `scripts/get_target.py` lists the eight increment commits explicitly: I5 is `19c7057` and I6 is `5cad1b2`, the first commit of each, which contains the generated code. Not increments: `2be6655` and `e1e3f63` (I5 follow-ups), `13680ed` (prep for I6), `807a50a` ("i6 done", no change under `app/studio/`).
- 2026-11-12: On my machine `TARGET_REPO_PATH` points to my existing checkout (`C:/Work/dance-studio-webapp`), so the script only fetches. With the default `data/target` it clones. Nothing reads or writes the working tree of the target.
- 2026-11-12: `GitRepo` differs from the plan's suggested git commands in two places. `diff` uses `git diff-tree -p --root` instead of `git diff <sha>^ <sha>`, because the latter fails on a commit without a parent. `read` uses `git cat-file blob` instead of `git show`, because `git show <sha>:<folder>` prints a folder listing instead of failing. Every method first resolves the SHA, so an unknown commit always raises `UnknownCommit`.
- 2026-11-12: Unit rules as implemented in `is_reviewable`: path starts with `app/studio/`, ends with `.py`, is not under `migrations/`, and the file name does not start with `seed_` (`seed_data.py`, `management/commands/seed_demo.py`). A unit needs at least 15 added lines as counted by `git diff-tree --numstat` against the parent commit; renames are not followed. Result: 26 dev units (I1: 9, I2: 8, I3: 9) and 36 test units (I4: 6, I5: 10, I6: 3, I7: 7, I8: 10), which matches the corrected specification. None of the increment commits changes a migration or seed file, so the exclusions do not remove anything at the moment.
- 2026-11-12: `.gitattributes` forces LF line endings for `data/*.json` and `data/*.jsonl`, so the checksum of the test set is the same on Windows and Linux.

## Problems

- 2026-11-12: `pip install -c <lab requirements URL> ...` ignored every pin and installed the newest versions (langchain 1.5.0, langgraph 1.2.20). Cause: my laptop is Windows on ARM running x64 Python, where `platform.machine()` returns `ARM64`, and every line of the lab file carries the marker `platform_machine == 'AMD64' and sys_platform == 'win32'`. Fix: evaluated the markers with `platform_machine` set to `AMD64`, wrote the matching lines to a plain constraints file without markers, and reinstalled into a fresh venv with that file. `pip check` reports no broken requirements. `requirements.txt` has no markers, so it installs the same versions on any machine.
