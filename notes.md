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
| 2026-11-12 | Claude Code | Exercise 1.3: checksum test in `tests/test_dataset.py`, first version of `DATA.md`, searched the target's history for committed secrets. |
| 2026-11-12 | Claude Code | Exercise 2.1: finding and result schemas and `clean_findings` in `schemas.py`, `tests/test_schemas.py`. |
| 2026-11-12 | Claude Code | Exercise 2.2: `llm.py` (model factory, `call_structured`, `add_usage`), `scripts/check_llm.py`, `tests/test_llm.py`. |

## Decisions

- 2026-11-12: There is no template repository, so the repository was created with `git init` in this folder. Work happens directly on `main` instead of a `solution` branch.
- 2026-11-12: Python 3.12.7 (x64) in `.venv`. `requirements.txt` is a `pip freeze` of an install constrained by the lab 3 requirements file, so shared packages have the lab versions: langchain 1.4.2, langchain-core 1.6.5, langgraph 1.2.12, langchain-ollama 1.1.0, pydantic 2.13.5, google-genai 2.25.0. Packages the lab file does not contain: langchain-google-genai 4.4.0 (no conflict with langchain-core 1.6.5), python-dotenv 1.2.4, pytest 9.2.0.
- 2026-11-12: The range `29949a3..e594bc8` has 12 commits for 8 increments. `scripts/get_target.py` lists the eight increment commits explicitly: I5 is `19c7057` and I6 is `5cad1b2`, the first commit of each, which contains the generated code. Not increments: `2be6655` and `e1e3f63` (I5 follow-ups), `13680ed` (prep for I6), `807a50a` ("i6 done", no change under `app/studio/`).
- 2026-11-12: On my machine `TARGET_REPO_PATH` points to my existing checkout (`C:/Work/dance-studio-webapp`), so the script only fetches. With the default `data/target` it clones. Nothing reads or writes the working tree of the target.
- 2026-11-12: `GitRepo` differs from the plan's suggested git commands in two places. `diff` uses `git diff-tree -p --root` instead of `git diff <sha>^ <sha>`, because the latter fails on a commit without a parent. `read` uses `git cat-file blob` instead of `git show`, because `git show <sha>:<folder>` prints a folder listing instead of failing. Every method first resolves the SHA, so an unknown commit always raises `UnknownCommit`.
- 2026-11-12: Unit rules as implemented in `is_reviewable`: path starts with `app/studio/`, ends with `.py`, is not under `migrations/`, and the file name does not start with `seed_` (`seed_data.py`, `management/commands/seed_demo.py`). A unit needs at least 15 added lines as counted by `git diff-tree --numstat` against the parent commit; renames are not followed. Result: 26 dev units (I1: 9, I2: 8, I3: 9) and 36 test units (I4: 6, I5: 10, I6: 3, I7: 7, I8: 10), which matches the corrected specification. None of the increment commits changes a migration or seed file, so the exclusions do not remove anything at the moment.
- 2026-11-12: `.gitattributes` forces LF line endings for `data/*.json` and `data/*.jsonl`, so the checksum of the test set is the same on Windows and Linux.
- 2026-11-12: Test set frozen in commit `ddc21c7`. SHA-256 of `data/units_test.jsonl`: `50e9c9dfd58b5f3c577414a4def7d423d4ae634db19e2563384703043526d5de`. From now on I do not open the diffs of test units; I only see them through the finished system.
- 2026-11-12: Dev units checked by file name and size (26 files, 15 to 218 added lines, files of 18 to 298 lines): all are application modules under `backend/`, `database/`, `system/`, `usecases/`, plus `forms.py`, `context_processors.py`, a template tag module and one management command. No generated or vendored file.
- 2026-11-12: Secrets in the target: none found in the whole history. `.env` was never committed; `.env.example` has placeholders; `app/dsw/settings.py` reads secrets with `env(...)`; passwords in `app/tests/` and `seed_demo.py` are demo values. For `is_readable()` in Part 3 a deny list is still worth having as a safeguard (`.env*`, `*.pem`, `*.key`), although nothing matches it today.
- 2026-11-12: `clean_findings(findings, file_text, path)` takes the reviewed path as a third argument (the plan lists two), because it has to set `file` to that path. Lines outside the file are moved to the nearest valid line instead of dropping the finding; whether the finding is right is for the self-check and the judge to decide.
- 2026-11-12: Retry layer for API errors: the chat model's own `max_retries=3` (Gemini), nothing else. When Part 4 adds a LangGraph `RetryPolicy` on the nodes, it must not retry API errors as well, or the attempts multiply. Ollama has no retry: a local connection error does not go away by waiting.
- 2026-11-12: `call_structured` returns `(parsed, raw_messages)` with one raw message per attempt, not a single raw message as in the plan. With a retry there are two LLM calls, and both have to be counted for the cost numbers. `StructuredOutputError` carries the raw messages for the same reason. `add_usage(stats, raw_messages)` does the counting.
- 2026-11-12: Checked on one Gemini call (`scripts/check_llm.py`, `gemini-3.8-flash`): `usage_metadata` was `input_tokens` 48, `output_tokens` 564, `output_token_details.reasoning` 447. So thinking tokens are included in `output_tokens` (117 visible + 447 thinking), and they were about 80% of the output here. Cost has to be computed from `output_tokens` as it is. Ollama (`qwen2.5-coder:7b`) reported 53 input and 95 output tokens for the same prompt.
- 2026-11-12: One rate limiter is shared by all Gemini models in the process (`get_rate_limiter` is cached), so reviewer and judge together stay under `REQUESTS_PER_MINUTE`.

## Problems

- 2026-11-12: `pip install -c <lab requirements URL> ...` ignored every pin and installed the newest versions (langchain 1.5.0, langgraph 1.2.20). Cause: my laptop is Windows on ARM running x64 Python, where `platform.machine()` returns `ARM64`, and every line of the lab file carries the marker `platform_machine == 'AMD64' and sys_platform == 'win32'`. Fix: evaluated the markers with `platform_machine` set to `AMD64`, wrote the matching lines to a plain constraints file without markers, and reinstalled into a fresh venv with that file. `pip check` reports no broken requirements. `requirements.txt` has no markers, so it installs the same versions on any machine.
