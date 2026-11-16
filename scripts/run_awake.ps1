# Runs `python -m reviewgraph <arguments>` and keeps Windows awake until it has finished.
#
# Usage, from the homework folder:
#   powershell -ExecutionPolicy Bypass -File scripts\run_awake.ps1 out\my-run.log review --repo <path> --commit <sha>
#
# The first argument is the log file; everything after it goes to reviewgraph.
# Normal output goes to the log file, error messages to "<log file>.err".
#
# No power setting is changed. The script only tells Windows "a job is running"
# for as long as it runs, so the idle timer does not put the laptop to sleep.
# Closing the lid or pressing the power button still does.

$log = $args[0]
$reviewgraphArguments = $args[1..($args.Count - 1)]

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
# 0x80000000 = "until I say otherwise", 0x00000001 = "the system is needed".
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

$env:PYTHONIOENCODING = "utf-8"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$process = Start-Process -FilePath $python `
    -ArgumentList (@("-u", "-m", "reviewgraph") + $reviewgraphArguments) `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
    -NoNewWindow -Wait -PassThru

Add-Content -Path $log -Value "exit code: $($process.ExitCode), finished $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
