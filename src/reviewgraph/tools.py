"""The read-only tools of the review agent, bound to one commit."""

from difflib import get_close_matches
from fnmatch import fnmatch

from langchain.tools import tool

from reviewgraph.gitrepo import GitError, GitRepo, PathNotFound, clean_path

MAX_LINES = 200
MAX_HITS = 40
MAX_FILES = 200
MAX_HIT_LENGTH = 200         # characters of one search hit
MAX_LINE_LENGTH = 1000       # characters of one line of a file

# Files that may hold secrets. The tools send what they read to the LLM provider
# and into the published traces, so these are never read, searched or listed.
DENIED_FILE_NAMES = (".env", ".env.*", "*.pem", "*.key", "*.p12", "id_rsa*")
ALLOWED_FILE_NAMES = (".env.example",)


def is_readable(path: str) -> bool:
    """False for files that may hold secrets, whatever folder they are in."""
    file_name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if file_name in ALLOWED_FILE_NAMES:
        return True
    return not any(fnmatch(file_name, pattern) for pattern in DENIED_FILE_NAMES)


def shorten(line: str, limit: int) -> str:
    """Cut very long lines (minified files), so one line cannot fill the context."""
    if len(line) <= limit:
        return line
    return line[:limit] + " [line cut]"


def make_tools(repo: GitRepo, sha: str) -> list:
    """Tools that see the repository at `sha`. They return text and never raise.

    The commit is fixed here and is not an argument of the tools, so the model
    cannot ask for a later version of the code.
    """
    sha = repo.resolve(sha)

    def suggest(path: str) -> str:
        """A "Did you mean ...?" sentence for a path that does not exist, or ""."""
        files = [file for file in repo.ls(sha) if is_readable(file)]
        folders = {file.rsplit("/", 1)[0] for file in files if "/" in file}
        matches = get_close_matches(path, files + sorted(folders), n=2, cutoff=0.9)
        return f" Did you mean {' or '.join(matches)}?" if matches else ""

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file of the project as it was at the reviewed commit.

        Pass the path from the project root with forward slashes, for example
        "app/studio/forms.py". start_line and end_line choose the lines to read;
        both are included and counted from 1. Leave end_line at 0 to read up to
        200 lines from start_line.

        Returns the requested lines, each with its line number in front, at most
        200 lines per call. The last line says if the file has more lines.
        """
        try:
            path = clean_path(path)
            if not is_readable(path):
                return f"Refused: {path} may contain secrets and cannot be read."
            if start_line < 1:
                return f"Invalid start_line {start_line}. Lines are counted from 1."
            if end_line == 0:
                end_line = start_line + MAX_LINES - 1
            if end_line < start_line:
                return f"Invalid line range {start_line}-{end_line}. end_line must not be smaller than start_line."
            try:
                lines = repo.read(sha, path).splitlines()
            except PathNotFound:
                if repo.ls(sha, path):
                    return f"{path} is a directory. Use list_files to see its files."
                return f"No such file at this commit: {path}.{suggest(path)}"

            if not lines:
                return f"{path} is empty."
            if start_line > len(lines):
                return f"{path} has only {len(lines)} lines; start_line {start_line} is after the end."

            end_line = min(end_line, start_line + MAX_LINES - 1, len(lines))
            numbered = [
                f"{number:>5}   {shorten(lines[number - 1], MAX_LINE_LENGTH)}" for number in range(start_line, end_line + 1)
            ]
            if end_line < len(lines):
                numbered.append(
                    f"[lines {start_line}-{end_line} of {len(lines)}; "
                    f"call again with start_line={end_line + 1} for more]"
                )
            return "\n".join(numbered)
        except PathNotFound as e:
            return f"Refused: {e}"
        except GitError as e:
            return f"Could not read {path}: {e}"

    @tool
    def search_code(pattern: str, directory: str = "app") -> str:
        """Search the project's code as it was at the reviewed commit.

        pattern is an extended regular expression (as in grep -E), for example
        "def cancel_booking" or "select_for_update|atomic". It is case sensitive.
        directory limits the search to one folder or one file, for example
        "app/studio/database".

        Returns up to 40 hits as `path:line: text`. The last line says if there
        were more hits.
        """
        try:
            if not pattern.strip():
                return "The search pattern is empty. Pass a word or a regular expression."
            directory = clean_path(directory)
            hits = []
            for hit in repo.grep(sha, pattern, directory):
                path, line_number, text = hit.split(":", 2)
                if is_readable(path):
                    hits.append(f"{path}:{line_number}: {shorten(text.strip(), MAX_HIT_LENGTH)}")

            if not hits:
                if not repo.ls(sha, directory):
                    return f"No such file or directory at this commit: {directory}.{suggest(directory)}"
                return f"No matches for {pattern!r} in {directory}."
            if len(hits) > MAX_HITS:
                shown = hits[:MAX_HITS]
                shown.append(f"[truncated, {MAX_HITS} of {len(hits)} hits; narrow the pattern or the directory]")
                return "\n".join(shown)
            return "\n".join(hits)
        except PathNotFound as e:
            return f"Refused: {e}"
        except GitError as e:
            # For an invalid regular expression this is git's own message.
            return f"Search failed: {e}"

    @tool
    def list_files(directory: str = "app/studio") -> str:
        """List the files below a directory of the project, as they were at the reviewed commit.

        Pass the directory from the project root with forward slashes, for
        example "app/studio/usecases". Files in subfolders are included.

        Returns one path per line, at most 200. The last line says if there
        were more files.
        """
        try:
            directory = clean_path(directory)
            files = [path for path in repo.ls(sha, directory) if is_readable(path)]
            if not files:
                return f"No such directory at this commit: {directory}.{suggest(directory)}"
            if len(files) > MAX_FILES:
                shown = files[:MAX_FILES]
                shown.append(f"[truncated, {MAX_FILES} of {len(files)} files; list a subfolder]")
                return "\n".join(shown)
            return "\n".join(files)
        except PathNotFound as e:
            return f"Refused: {e}"
        except GitError as e:
            return f"Could not list {directory}: {e}"

    return [read_file, search_code, list_files]
