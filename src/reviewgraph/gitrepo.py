import re
import subprocess
from pathlib import Path

from pydantic import BaseModel

SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{4,40}$")
GIT_TIMEOUT_SECONDS = 30


class GitError(Exception): ...
class UnknownCommit(GitError): ...
class PathNotFound(GitError): ...


class ChangedFile(BaseModel):
    path: str
    added: int
    deleted: int


def clean_path(path: str) -> str:
    """Return `path` with forward slashes; PathNotFound if it could leave the repository."""
    cleaned = path.strip().replace("\\", "/")
    is_absolute = cleaned.startswith("/") or re.match(r"^[A-Za-z]:", cleaned)
    if not cleaned or is_absolute or ".." in cleaned.split("/"):
        raise PathNotFound(f"Not a path inside the repository: {path!r}")
    return cleaned.removeprefix("./")


class GitRepo:
    """Read-only view of a repository. Never reads or writes the working tree."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise GitError(f"Not a directory: {self.path}")
        try:
            self._git("rev-parse", "--git-dir")
        except GitError:
            raise GitError(f"Not a git repository: {self.path}") from None

    def _git(self, *args: str, ok_codes: tuple[int, ...] = (0,)) -> str:
        """Run `git -C self.path ...` and return stdout as text."""
        command = ["git", "--no-optional-locks", "-C", str(self.path), *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            raise GitError("git is not installed or not on the PATH.") from None
        except subprocess.TimeoutExpired:
            raise GitError(f"git {args[0]} timed out after {GIT_TIMEOUT_SECONDS} s.") from None
        if result.returncode not in ok_codes:
            raise GitError(f"git {args[0]} failed: {result.stderr.strip()}")
        return result.stdout

    def resolve(self, rev: str) -> str:
        """Full SHA of `rev`; UnknownCommit if it does not name a commit."""
        # Only hex SHAs reach git: a value like "--output=x" would be read as an option.
        if not SHA_PATTERN.match(rev):
            raise UnknownCommit(f"Not a commit SHA: {rev!r}")
        try:
            return self._git("rev-parse", "--verify", "--quiet", rev + "^{commit}").strip()
        except GitError:
            raise UnknownCommit(f"Unknown commit: {rev}") from None

    def changed_files(self, sha: str) -> list[ChangedFile]:
        """Files changed by the commit, compared with its parent."""
        sha = self.resolve(sha)
        # --root makes the first commit of a repository list all its files.
        # -z separates the records with NUL, so unusual file names are not quoted.
        output = self._git("diff-tree", "--no-commit-id", "--numstat", "-r", "--root", "-z", sha)
        files = []
        for record in output.split("\0"):
            if not record:
                continue
            added, deleted, path = record.split("\t", 2)
            # Binary files have "-" instead of line counts.
            files.append(
                ChangedFile(
                    path=path,
                    added=int(added) if added.isdigit() else 0,
                    deleted=int(deleted) if deleted.isdigit() else 0,
                )
            )
        return files

    def diff(self, sha: str, path: str, context: int = 10) -> str:
        """Unified diff of one file in the commit; empty if the commit did not change it."""
        sha = self.resolve(sha)
        path = clean_path(path)
        # diff-tree is used instead of `git diff <sha>^ <sha>`, because with --root
        # it also works for the first commit, which has no parent.
        return self._git(
            "diff-tree", "--no-commit-id", "-p", "--root", f"-U{context}", sha, "--", path
        )

    def read(self, sha: str, path: str) -> str:
        """Content of the file as it was at the commit."""
        sha = self.resolve(sha)
        path = clean_path(path)
        try:
            # `cat-file blob` fails for a missing path and for a folder,
            # while `git show` would list the folder.
            return self._git("cat-file", "blob", f"{sha}:{path}")
        except GitError:
            raise PathNotFound(f"No file {path} at commit {sha[:7]}") from None

    def ls(self, sha: str, prefix: str = "") -> list[str]:
        """Paths of all files at the commit, optionally only under `prefix`."""
        sha = self.resolve(sha)
        args = ["ls-tree", "-r", "--name-only", "-z", sha]
        if prefix:
            args += ["--", clean_path(prefix)]
        return [path for path in self._git(*args).split("\0") if path]

    def grep(self, sha: str, pattern: str, pathspec: str = "app") -> list[str]:
        """Lines matching the regular expression at the commit, as "path:line:text"."""
        sha = self.resolve(sha)
        pathspec = clean_path(pathspec)
        if not pattern:
            raise GitError("The search pattern is empty.")
        # Exit code 1 means "no matches", not an error.
        output = self._git(
            "grep", "-n", "-I", "-E", "--no-color", "-e", pattern, sha, "--", pathspec,
            ok_codes=(0, 1),
        )
        # git prefixes every line with "<sha>:".
        return [line.removeprefix(sha + ":") for line in output.splitlines()]
