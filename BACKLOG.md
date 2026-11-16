# Backlog

Things that are known and not done, each with the reason. When one is done, move the result to `notes.md` and delete the entry here.

## From Exercise 5.3 (error cases)

### An unexpected error still ends with a traceback

- **What:** `main` in `cli.py` catches `GitError`, `ConfigError` and `OllamaError` only. Any other exception that escapes (a bug in the code) prints a Python traceback instead of one line.
- **Why not done:** on purpose. These three are the mistakes a user can make and fix. Anything else is a bug, and the traceback is what is needed to find it. A catch-all `except Exception` would hide the place where it happened.
- **If it is done later:** catch everything in `main`, print one line, and show the traceback only with `--verbose`.

### A real Gemini failure has not been run through the command line

- **What:** the rows "API error or rate limit" and "structured output invalid twice" are tested with the scripted model only. The same is true for a wrong API key. What a real Gemini error looks like in the `FAILED:` line and in the report has not been seen.
- **Why not done:** the free tier allows 20 requests per day and they were used up on 2026-11-16. Every failed request counts against the quota as well, so a run that fails on purpose costs requests that the open checkpoints need.
- **If it is done later:** one run of `reviewgraph review` on a commit with a single small file and a broken `GEMINI_API_KEY` costs one request per file. It can be combined with the open 4.4 checkpoint, which also asks for a run with a broken key.

### Ctrl+C loses the files reviewed so far

- **What:** the reports are written once, after the last file. If a run is stopped with Ctrl+C (or the laptop goes to sleep and the run is killed), the results of the files that were already reviewed are not written anywhere.
- **Why not done:** it is not in the error table of Exercise 5.3, and the proper answer is already planned: the resumable runner of Exercise 6.1 writes one file per unit and skips the units that exist. Solving it a second time in the command line would be duplicate code.
- **Why it still matters:** a review with a local model takes hours, so a stopped run is expensive. Until then: let a run finish, or use `--verbose` with the output sent to a log file, where at least the progress lines survive.
- **If it is done later:** catch `KeyboardInterrupt` in `run_review`, write the reports for the results so far, and exit with 3.
