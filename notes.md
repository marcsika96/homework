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
| 2026-11-12 | Claude Code | Exercise 2.3: `prompts.py` (system prompt, `render_diff`, `load_rules`), `baseline.py`, their tests; found that `gemini-3.8-flash` was unavailable and compared the other Gemini models. |
| 2026-11-12 | Claude Code | Exercise 3.1: `tools.py` with `read_file`, `search_code`, `list_files` and `is_readable`; `scripts/try_baseline.py`; first tests in `tests/test_tools.py`. |
| 2026-11-12 | Ollama (`qwen2.5-coder:7b`) | Ran the baseline on one dev unit with `scripts/try_baseline.py` to try the key-free path. |
| 2026-11-12 | Claude Code | Exercise 3.2: hardening tests for the tools in `tests/test_tools.py`; binary files are now reported instead of shown. |
| 2026-11-12 | Claude Code | Exercise 4.1: reviewed my load_context (found two calls with missing arguments), explained how to run a single node. |


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
- 2026-11-12: `review_baseline` never raises for a failed review. An unknown commit or a file the commit does not change is reported in `result.error` without an LLM call; an API failure and invalid output are reported there too, with the calls counted. Only a configuration error (missing key) is raised.
- 2026-11-12: `latency_s` is the wall-clock time of the whole review. It includes the waiting time of the rate limiter and the model's own retries, so the latency numbers of the evaluation have to say so, or the limiter wait has to be measured separately.
- 2026-11-12: Gemini requests have a timeout of 300 s (`API_TIMEOUT_SECONDS`). Without a timeout a request hung for more than five minutes. One review of a 298-line file took 110 s on `gemini-3.5-flash`, almost all of it thinking.
- 2026-11-12: The Gemini model is `gemini-3.5-flash` everywhere (default in `config.py`, `.env.example`, specification), not `gemini-3.8-flash` as in lab 3. Reason: 3.8 was overloaded (503/504) on 2026-11-09 and again on 2026-11-12, while 3.5 answered and honours temperature 0. Older entries in this file that name `gemini-3.8-flash` describe what was run at the time.
- 2026-11-12: Tool limits: `read_file` returns at most 200 lines per call and cuts lines after 1,000 characters; `search_code` returns at most 40 hits, each cut after 200 characters (the vendored minified JavaScript under `app/` has lines of many thousand characters); `list_files` returns at most 200 paths. Cut output always ends with a line that says so and what to do.
- 2026-11-12: `read_file` has `end_line: int = 0`, where 0 means "up to 200 lines from `start_line`" (the plan has `end_line = MAX_LINES`). With a fixed default of 200, a call with only `start_line=250` would be an invalid range.
- 2026-11-12: `is_readable` deny list (file name, in any folder): `.env`, `.env.*` except `.env.example`, `*.pem`, `*.key`, `*.p12`, `id_rsa*`. Denied files are not read, not shown in search hits and not listed. Nothing in the target matches today.

## The reviewed code is untrusted input (for the README architecture section)

The code under review is written by someone else (here: by AI agents), and a comment or string in it could tell the model to ignore its rules. The tools limit what such an injection can achieve: they can only read tracked files of one fixed commit, so the model cannot change or run anything, cannot leave the repository, cannot read files on the deny list, and cannot reach code from a later commit. The worst outcome is a wrong or missing finding in the report, which the self-check and the developer reading the report can still catch.

Tested in `tests/test_tools.py` on the fixture repository, without network or the cloned target: paths outside the repository, files from a later commit, the working tree, invalid regular expressions, bad line ranges, deny-listed files, output limits, and arguments that look like git options.

## Baseline checkpoint (dev units, prompt version 1, `gemini-3.5-flash`)

| Unit | Findings | Latency | Input tokens | Output tokens |
|------|----------|---------|--------------|---------------|
| `I1:app/studio/system/transactions.py` | 0 | 21.6 s | 4,577 | 1,795 |
| `I3:app/studio/backend/classes.py` | 0 | 110.3 s | 7,374 | 5,315 |
| `I3:app/studio/database/classes.py` | 1 | 64.2 s | 6,306 | 4,061 |

The one finding (lines 52–54, `logic`, high): `ends_at` adds a duration to the `starts_at` datetime but declares `output_field=DurationField()`; the fix is `DateTimeField()`. It is inside the file and on added lines. I have not yet confirmed it by running the code.

- 2026-11-12: Same unit and prompt (version 1) on the local model: `I3:app/studio/database/classes.py` with `qwen2.5-coder:7b` through Ollama took 558.5 s, used 5,984 input and 8 output tokens, and returned no findings. `gemini-3.5-flash` took 64.2 s on this unit and reported the `DurationField` defect at lines 52–54. The free path works, but on my laptop it is about nine times slower and missed the defect, so the evaluation runs on Gemini and Ollama stays the key-free fallback.


## Problems

- 2026-11-12: `gemini-3.8-flash` was not usable in the afternoon: `503 UNAVAILABLE` ("model is currently experiencing high demand") and `504 DEADLINE_EXCEEDED`, even for a 48-token request. The same review request sent to the other models at 15:40: `gemini-3.8-flash` 504 after 94 s; `gemini-3.7-flash` 503; `gemini-3.6-flash` 503 (and it ignores `temperature`); `gemini-2.5-flash` 404, no longer available; `gemini-3.5-flash` OK in 24 s; `gemini-3.5-flash-lite` OK in 6.5 s, but it ignores `temperature` and does not think. The checkpoint was run with `GEMINI_MODEL=gemini-3.5-flash`.
- 2026-11-12: `pip install -c <lab requirements URL> ...` ignored every pin and installed the newest versions (langchain 1.5.0, langgraph 1.2.20). Cause: my laptop is Windows on ARM running x64 Python, where `platform.machine()` returns `ARM64`, and every line of the lab file carries the marker `platform_machine == 'AMD64' and sys_platform == 'win32'`. Fix: evaluated the markers with `platform_machine` set to `AMD64`, wrote the matching lines to a plain constraints file without markers, and reinstalled into a fresh venv with that file. `pip check` reports no broken requirements. `requirements.txt` has no markers, so it installs the same versions on any machine.
