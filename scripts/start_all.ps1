# Start everything Swing Coach needs, in dependency order. Safe to run twice: each step
# is skipped when its port is already listening. Registered to run at Windows logon by
# scripts/install_autostart.ps1. Logs go to data\autostart_*.txt.
#
#   Ollama (11434) -> platform in WSL (Postgres 5432, Kafka 9094, Connect 8083)
#   -> app (8000, needs Postgres) -> CDC stream (needs Kafka + Connect)
#
# Keep this file ASCII: Windows PowerShell 5.1 misreads non-ASCII in BOM-less scripts,
# and the project path contains Hangul, so it is derived from $PSScriptRoot instead.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Data = Join-Path $Root "data"
New-Item -ItemType Directory -Force $Data | Out-Null
$Log = Join-Path $Data "autostart_log.txt"
$env:PYTHONIOENCODING = "utf-8"

function Say($msg) { "$(Get-Date -Format s) $msg" | Out-File -Append -Encoding utf8 $Log }

function Listening($port) {
    [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function WaitFor($ports, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (-not ($ports | Where-Object { -not (Listening $_) })) { return $true }
        Start-Sleep 3
    }
    Say "timeout waiting for ports $($ports -join ',')"
    return $false
}

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in "$env:USERPROFILE\.local\bin\uv.exe", "$env:LOCALAPPDATA\hermes\bin\uv.exe") {
        if (Test-Path $p) { return $p }
    }
    throw "uv not found"
}

function StartHidden($name, $file, $arguments) {
    Start-Process -FilePath $file -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $Data "autostart_$name.txt") `
        -RedirectStandardError (Join-Path $Data "autostart_$name.err.txt")
    Say "started $name"
}

Say "---- start"
try {
    $uv = Find-Uv

    # 1. Local LLM (optional: the app falls back to rule-based text without it)
    $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (-not (Listening 11434) -and (Test-Path $ollama)) { Start-Process $ollama; Say "started ollama" }

    # 2. Platform. Compose stays attached on purpose: WSL shuts down when no process holds it.
    if (-not (Listening 5432)) {
        $wslDir = "/mnt/" + $Root.Substring(0, 1).ToLower() + ($Root.Substring(2) -replace "\\", "/") + "/platform"
        $compose = "cd '$wslDir' && exec sg docker -c 'docker compose --env-file .env up --no-color'"
        StartHidden "platform" "wsl.exe" @("-d", "Ubuntu-24.04", "--", "bash", "-lc", "`"$compose`"")
    }

    # 3. App (its job queue lives in Postgres). WSL's relay opens 5432 before Postgres is
    # ready, so the app can exit on a connect timeout: retry until it is listening.
    if (WaitFor @(5432) 240) {
        for ($try = 1; $try -le 5 -and -not (Listening 8000); $try++) {
            $app = Start-Process -FilePath $uv -ArgumentList @("run", "--no-sync", "python", "scripts/serve.py") `
                -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $Data "autostart_app.txt") `
                -RedirectStandardError (Join-Path $Data "autostart_app.err.txt")
            Say "started app (try $try)"
            $deadline = (Get-Date).AddSeconds(90)
            while ((Get-Date) -lt $deadline -and -not $app.HasExited -and -not (Listening 8000)) { Start-Sleep 3 }
            if (-not (Listening 8000)) {
                if (-not $app.HasExited) { & taskkill.exe /T /F /PID $app.Id | Out-Null }
                Start-Sleep 10
            }
        }
    }

    # 4. CDC stream (Kafka broker + Debezium Connect)
    if (WaitFor @(9094, 8083) 240) {
        $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -match "pipeline[\\/]stream\.py" }
        if (-not $running) { StartHidden "stream" $uv @("run", "--no-sync", "python", "pipeline/stream.py") }
    }
    Say "done"
} catch {
    Say "error: $($_.Exception.Message)"
    throw
}
