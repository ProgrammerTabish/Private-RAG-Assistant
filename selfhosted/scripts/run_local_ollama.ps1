<#
.SYNOPSIS
  Local end-to-end test of the self-hosted pipeline with the REAL BGE-M3 embedding model
  (and a small open-source LLM) served by Ollama - the same OpenAI-compatible endpoints
  the pipeline will call on Azure (TEI /v1/embeddings, vLLM /v1/chat/completions).

.EXAMPLE
  cd D:\Private-RAG-Assistant
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\run_local_ollama.ps1

.EXAMPLE
  # quick smoke test: 5 questions only
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\run_local_ollama.ps1 -Limit 5

.NOTES
  Results:  selfhosted\data_ollama\reports\  (eval_*.json / .xlsx, local_run_info.json)
  Logs:     selfhosted\logs\  (JSON lines per run + local-ollama-<time>.txt transcript)
  Re-running is safe: unchanged PDFs and an unchanged index are reused.
#>
[CmdletBinding()]
param(
    [string]$LlmModel = "auto",         # auto | qwen2.5:7b | qwen2.5:3b | llama3.1:8b | none (= fake LLM)
    [int]$Limit = 0,                    # only the first N questions (0 = all 32)
    [int]$TopK = 4,                     # chunks given to the LLM (Azure: 8 - lower here for CPU speed)
    [int]$MaxSourceChars = 900,         # characters per source in the prompt (Azure: 1800)
    [int]$ContextTokens = 4096,         # Ollama context window for the LLM
    [int]$MaxAnswerTokens = 350,        # answer length limit (Azure: 900)
    [switch]$QueryRewrite,              # extra LLM call for German search terms (Azure: on; slow on a CPU)
    [int]$EmbedMinutes = 5,             # BGE-M3 time budget per index run; all chunks always get keyword search
    [switch]$Improve,                   # on later starts: spend another -EmbedMinutes on more BGE-M3 coverage
    [switch]$Full,                      # no time budget: embed everything (hours on a laptop CPU)
    [int]$EmbedThreads = 0,             # CPU threads for BGE-M3 (0 = all logical processors)
    [switch]$SkipIndex,                 # reuse the existing index (data_ollama)
    [switch]$NoEval,                    # build only (no 32-question evaluation)
    [switch]$Serve,                     # start the API + chat UI at the end and open the browser
    [int]$Port = 8000,
    [string]$QdrantVersion = "v1.19.2",
    [string]$OllamaUrl = "http://127.0.0.1:11434"
)

$ErrorActionPreference = "Stop"
$SelfHosted = Split-Path -Parent $PSScriptRoot
Set-Location $SelfHosted
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force -Path (Join-Path $SelfHosted "logs") | Out-Null
$Transcript = Join-Path $SelfHosted "logs\local-ollama-$Stamp.txt"
Start-Transcript -Path $Transcript | Out-Null
$Info = [ordered]@{ started = (Get-Date).ToString("s"); steps = [ordered]@{} }
$Total = [Diagnostics.Stopwatch]::StartNew()

function Step([string]$msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail([string]$msg) {
    Write-Host ""; Write-Host "FAILED: $msg" -ForegroundColor Red
    Write-Host "Transcript: $Transcript" -ForegroundColor Red
    Write-Host "Last errors: .venv-win\Scripts\python.exe -m privrag.cli logs --errors" -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
# Native programs (ollama, python) write progress to stderr; in Windows PowerShell 5.1 that must
# not be treated as a script error, so we only look at the exit code.
function Invoke-Native([string]$what, [scriptblock]$block) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $block
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    $Info.steps[$what] = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    if ($code -ne 0) { Fail "$what (exit code $code)" }
    Write-Host ("    {0}: done in {1:N0} s" -f $what, $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
}

try {
    # ------------------------------------------------------------------ 1 machine
    Step "1/7 Machine"
    $cpuName = "unknown"; $cores = 0; $threads = [Environment]::ProcessorCount; $ramGB = 0; $freeGB = 0
    try {
        $cs = Get-CimInstance Win32_ComputerSystem
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        $cpuName = $cpu.Name; $cores = $cpu.NumberOfCores; $threads = $cpu.NumberOfLogicalProcessors
        $ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
        $freeGB = [math]::Round((Get-PSDrive -Name (Split-Path -Qualifier $SelfHosted).TrimEnd(':')).Free / 1GB, 1)
    } catch { Write-Host "(machine details unavailable: $($_.Exception.Message))" -ForegroundColor DarkGray }
    Write-Host "CPU: $cpuName  ($cores cores / $threads threads)"
    Write-Host "RAM: $ramGB GB   free disk on project drive: $freeGB GB"
    if ($freeGB -gt 0 -and $freeGB -lt 3) { Fail "less than 3 GB free on the project drive" }
    if ($LlmModel -eq "auto") {
        # On a laptop CPU a 7B model needs ~2-4 min per question (~2 h for 32); 3B ~1-1.5 min.
        # Use -LlmModel qwen2.5:7b for better answers if you have the time and >= 16 GB RAM.
        $LlmModel = "qwen2.5:3b"
        Write-Host "LLM: $LlmModel (pass -LlmModel qwen2.5:7b for a bigger, slower model)"
    }
    $Info.machine = [ordered]@{ cpu = $cpuName; cores = $cores; threads = $threads; ram_gb = $ramGB }

    # ------------------------------------------------------------------ 2 ollama + models
    Step "2/7 Ollama and models"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Fail "ollama not found. Install it from https://ollama.com/download and run this script again."
    }
    $up = $false
    try { $v = Invoke-RestMethod "$OllamaUrl/api/version" -TimeoutSec 5; $up = $true } catch {}
    if (-not $up) {
        Write-Host "Ollama is not running - starting 'ollama serve' in the background ..."
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 30 -and -not $up; $i++) {
            Start-Sleep -Seconds 2
            try { $v = Invoke-RestMethod "$OllamaUrl/api/version" -TimeoutSec 5; $up = $true } catch {}
        }
        if (-not $up) { Fail "Ollama did not start on $OllamaUrl" }
    }
    Write-Host "Ollama version $($v.version) at $OllamaUrl"
    $Info.ollama_version = $v.version

    Invoke-Native "pull bge-m3" { ollama pull bge-m3 }
    # Same BGE-M3 weights, but Ollama by default uses only half of the cores for it.
    # Using all of them is the biggest speed-up available on a CPU-only laptop.
    if ($EmbedThreads -le 0) { $EmbedThreads = [Environment]::ProcessorCount }
    $emf = Join-Path ([IO.Path]::GetTempPath()) "privrag-embed.Modelfile"
    "FROM bge-m3`nPARAMETER num_thread $EmbedThreads`n" | Set-Content -Path $emf -Encoding ascii
    Invoke-Native "create privrag-embed" { ollama create privrag-embed -f $emf }
    $EmbedModel = "privrag-embed"
    Write-Host "Embedding model: bge-m3 weights as '$EmbedModel' with $EmbedThreads CPU threads"
    $LlmServed = "none"
    if ($LlmModel -ne "none") {
        Invoke-Native "pull $LlmModel" { ollama pull $LlmModel }
        # vLLM on Azure gets max_model_len; Ollama needs num_ctx in a Modelfile, otherwise it silently
        # truncates long RAG prompts (default context is only 2-4k tokens).
        $mf = Join-Path ([IO.Path]::GetTempPath()) "privrag.Modelfile"
        "FROM $LlmModel`nPARAMETER num_ctx $ContextTokens`nPARAMETER temperature 0`n" |
            Set-Content -Path $mf -Encoding ascii
        Invoke-Native "create privrag-llm" { ollama create privrag-llm -f $mf }
        $LlmServed = "privrag-llm"
    }
    $Info.models = [ordered]@{ embedding = "bge-m3 (Ollama, $EmbedThreads threads)"; llm = $LlmModel; llm_context = $ContextTokens }

    # ------------------------------------------------------------------ 3 endpoint smoke tests
    Step "3/7 Endpoint smoke tests (same API contract as Azure TEI / vLLM)"
    $body = @{ model = $EmbedModel; input = @("Meldepflicht nach § 43 GwG", "reporting obligation money laundering") } | ConvertTo-Json
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $emb = Invoke-RestMethod -Method Post "$OllamaUrl/v1/embeddings" -ContentType "application/json; charset=utf-8" `
        -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 300
    $dim = $emb.data[0].embedding.Count
    Write-Host ("/v1/embeddings OK: {0} vectors, dim {1}, {2:N2} s" -f $emb.data.Count, $dim, $sw.Elapsed.TotalSeconds)
    if ($dim -ne 1024) { Fail "bge-m3 returned dim $dim (expected 1024) - is the Ollama model correct?" }
    if ($LlmServed -ne "none") {
        $chat = @{ model = $LlmServed; messages = @(@{ role = "user"; content = "Antworte nur mit: OK" }); max_tokens = 5 } | ConvertTo-Json -Depth 4
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod -Method Post "$OllamaUrl/v1/chat/completions" -ContentType "application/json; charset=utf-8" `
            -Body ([Text.Encoding]::UTF8.GetBytes($chat)) -TimeoutSec 600
        Write-Host ("/v1/chat/completions OK: '{0}' in {1:N1} s (includes model load)" -f $r.choices[0].message.content.Trim(), $sw.Elapsed.TotalSeconds)
        # unload the chat model again so indexing has the RAM (it reloads on the first question)
        try {
            $unload = @{ model = $LlmServed; keep_alive = 0 } | ConvertTo-Json
            Invoke-RestMethod -Method Post "$OllamaUrl/api/generate" -ContentType "application/json" -Body $unload -TimeoutSec 60 | Out-Null
            Write-Host "Chat model unloaded until it is needed (frees RAM for indexing)"
        } catch { Write-Host "(could not unload chat model: $($_.Exception.Message))" -ForegroundColor DarkGray }
    }

    # ------------------------------------------------------------------ 3b qdrant server (as on Azure)
    Step "3b/7 Qdrant vector database server ($QdrantVersion, localhost only)"
    $QdrantUrl = "http://127.0.0.1:6333"
    $qUp = $false
    try { Invoke-RestMethod "$QdrantUrl/readyz" -TimeoutSec 3 | Out-Null; $qUp = $true } catch {}
    if ($qUp) {
        Write-Host "Qdrant already running at $QdrantUrl"
    } else {
        $qDir = Join-Path $SelfHosted "tools\qdrant-$QdrantVersion"
        $qExe = Join-Path $qDir "qdrant.exe"
        if (-not (Test-Path $qExe)) {
            New-Item -ItemType Directory -Force -Path $qDir | Out-Null
            $zip = Join-Path $qDir "qdrant.zip"
            $url = "https://github.com/qdrant/qdrant/releases/download/$QdrantVersion/qdrant-x86_64-pc-windows-msvc.zip"
            Write-Host "Downloading $url"
            $ProgressPreference = "SilentlyContinue"
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
            Expand-Archive -Path $zip -DestinationPath $qDir -Force
            Remove-Item $zip
            if (-not (Test-Path $qExe)) { $found = Get-ChildItem $qDir -Recurse -Filter qdrant.exe | Select-Object -First 1; if ($found) { $qExe = $found.FullName } }
            if (-not (Test-Path $qExe)) { Fail "qdrant.exe not found after download" }
        }
        $qData = Join-Path $SelfHosted "data_ollama\qdrant_server"
        New-Item -ItemType Directory -Force -Path $qData | Out-Null
        $env:QDRANT__SERVICE__HOST = "127.0.0.1"          # no network exposure, no firewall prompt
        $env:QDRANT__SERVICE__HTTP_PORT = "6333"
        $env:QDRANT__SERVICE__GRPC_PORT = "6334"
        $env:QDRANT__STORAGE__STORAGE_PATH = Join-Path $qData "storage"
        $env:QDRANT__STORAGE__SNAPSHOTS_PATH = Join-Path $qData "snapshots"
        $env:QDRANT__TELEMETRY_DISABLED = "true"
        $qLog = Join-Path $SelfHosted "logs\qdrant-$Stamp.log"
        Start-Process -FilePath $qExe -WorkingDirectory $qData -WindowStyle Hidden -RedirectStandardOutput $qLog -RedirectStandardError "$qLog.err"
        for ($i = 0; $i -lt 30 -and -not $qUp; $i++) {
            Start-Sleep -Seconds 1
            try { Invoke-RestMethod "$QdrantUrl/readyz" -TimeoutSec 3 | Out-Null; $qUp = $true } catch {}
        }
        if (-not $qUp) { Fail "Qdrant did not start - see $qLog.err" }
        Write-Host "Qdrant started at $QdrantUrl (data: $qData)"
    }
    $Info.qdrant = $QdrantVersion

    # ------------------------------------------------------------------ 4 python env
    Step "4/7 Python environment (.venv-win)"
    $Py = if ($IsLinux -or $IsMacOS) { Join-Path $SelfHosted ".venv-win/bin/python" } else { Join-Path $SelfHosted ".venv-win\Scripts\python.exe" }
    if (-not (Test-Path $Py)) {
        $cands = @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3.13"), @("py", "-3.10"), @("python"), @("python3"))
        $base = $null
        foreach ($c in $cands) {
            $exe = $c[0]; $rest = @($c | Select-Object -Skip 1)
            if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
            $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
            $ok = & $exe @rest -c "import sys; print(sys.version_info >= (3, 10))" 2>$null
            $ErrorActionPreference = $old
            if ($ok -eq "True") { $base = $c; break }
        }
        if (-not $base) { Fail "No Python >= 3.10 found. Install Python 3.12 from python.org (tick 'Add to PATH')." }
        Write-Host "Creating venv with: $($base -join ' ')"
        $exe = $base[0]; $rest = @($base | Select-Object -Skip 1)
        Invoke-Native "create venv" { & $exe @rest -m venv .venv-win }
    }
    Invoke-Native "pip install" { & $Py -m pip install --disable-pip-version-check -q --upgrade pip; & $Py -m pip install --disable-pip-version-check -q -e . }
    $Info.python = (& $Py -c "import sys; print(sys.version.split()[0])")

    # ------------------------------------------------------------------ 5 settings
    Step "5/7 Settings (local copy of the Azure configuration)"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PRIVRAG_ENV = "local"
    $env:PRIVRAG_DATA_DIR = Join-Path $SelfHosted "data_ollama"     # keeps the stand-in results in data\ untouched
    $env:PRIVRAG_LOG_CONSOLE = "true"
    $env:PRIVRAG_PARSER_BACKEND = "pymupdf"
    $env:PRIVRAG_EMBED_BACKEND = "remote"
    $env:PRIVRAG_EMBED_URL = "$OllamaUrl/v1"
    $env:PRIVRAG_EMBED_MODEL = $EmbedModel
    $env:PRIVRAG_EMBED_BATCH_SIZE = "8"           # work is saved every 64 chunks
    $env:PRIVRAG_EMBED_TIMEOUT_S = "600"
    $env:PRIVRAG_QDRANT_MODE = "server"
    $env:PRIVRAG_QDRANT_URL = $QdrantUrl
    $env:PRIVRAG_RERANK_BACKEND = "none"
    $env:PRIVRAG_TOP_K = "$TopK"
    $env:PRIVRAG_QUERY_REWRITE = if ($QueryRewrite) { "true" } else { "false" }
    # Time budget for BGE-M3: first build -EmbedMinutes, later starts 0 (instant) unless -Improve; -Full = no limit
    $metaFile = Join-Path $SelfHosted "data_ollama\index_meta.server.spg_compliance.json"
    if ($Full) {
        Remove-Item env:PRIVRAG_INDEX_TIME_BUDGET_S -ErrorAction SilentlyContinue
        $budgetTxt = "no limit (-Full)"
    } elseif ((Test-Path $metaFile) -and -not $Improve) {
        $env:PRIVRAG_INDEX_TIME_BUDGET_S = "0"
        $budgetTxt = "0 (index exists - quick start; use -Improve to add BGE-M3 coverage)"
    } else {
        $env:PRIVRAG_INDEX_TIME_BUDGET_S = "$($EmbedMinutes * 60)"
        $budgetTxt = "$EmbedMinutes min"
    }
    $env:PRIVRAG_MAX_SOURCE_CHARS = "$MaxSourceChars"
    $env:PRIVRAG_MIN_CITATIONS = "2"
    if ($LlmServed -ne "none") {
        $env:PRIVRAG_LLM_BACKEND = "openai"
        $env:PRIVRAG_LLM_BASE_URL = "$OllamaUrl/v1"
        $env:PRIVRAG_LLM_MODEL = $LlmServed
        $env:PRIVRAG_LLM_TIMEOUT_S = "900"
        $env:PRIVRAG_LLM_MAX_RETRIES = "1"
        $env:PRIVRAG_LLM_MAX_TOKENS = "$MaxAnswerTokens"
    } else {
        $env:PRIVRAG_LLM_BACKEND = "fake"
    }
    Get-ChildItem env:PRIVRAG_* | Sort-Object Name | ForEach-Object { Write-Host ("    {0} = {1}" -f $_.Name, $_.Value) }
    $Info.settings = [ordered]@{}
    Get-ChildItem env:PRIVRAG_* | ForEach-Object { $Info.settings[$_.Name] = $_.Value }

    # ------------------------------------------------------------------ 6 ingest + index
    Step "6/7 Ingest + index (BGE-M3 time budget: $budgetTxt)"
    Write-Host "All chunks get keyword (BM25) search in seconds; BGE-M3 vectors are added for as many chunks"
    Write-Host "as fit in the budget, spread over all 50 documents. Progress is printed every ~15 s."
    Write-Host "More coverage later: run again with -Improve (adds $EmbedMinutes min) or -Full (everything, hours)."
    Invoke-Native "doctor" { & $Py -m privrag.cli doctor --before-build }
    Invoke-Native "ingest" { & $Py -m privrag.cli ingest }
    if (-not $SkipIndex) { Invoke-Native "index" { & $Py -m privrag.cli index } }

    # ------------------------------------------------------------------ 7 eval
    if ($NoEval) {
        Step "7/7 Evaluation skipped (-NoEval)"
    } else {
        $n = if ($Limit -gt 0) { $Limit } else { "all 32" }
        Step "7/7 Evaluation ($n questions, LLM: $LlmModel)"
        if ($Limit -gt 0) {
            Invoke-Native "eval" { & $Py -m privrag.cli eval --limit $Limit }
        } else {
            Invoke-Native "eval" { & $Py -m privrag.cli eval }
        }
    }

    $Info.total_minutes = [math]::Round($Total.Elapsed.TotalMinutes, 1)
    $Info.finished = (Get-Date).ToString("s")
    $infoPath = Join-Path $SelfHosted "data_ollama\reports\local_run_info.json"
    $Info | ConvertTo-Json -Depth 5 | Set-Content -Path $infoPath -Encoding utf8
    Write-Host ""
    Write-Host ("DONE in {0:N1} min. Results: {1}" -f $Total.Elapsed.TotalMinutes, (Join-Path $SelfHosted "data_ollama\reports")) -ForegroundColor Green
    if (-not $Serve) {
        Write-Host "Tell Claude it has finished - the report is written from these files." -ForegroundColor Green
    } else {
        Step "Chat UI on http://127.0.0.1:$Port  (Ctrl+C in this window stops it)"
        $api = Start-Process -FilePath $Py -ArgumentList @("-m", "privrag.cli", "serve", "--host", "127.0.0.1", "--port", "$Port") `
            -NoNewWindow -PassThru
        $ok = $false
        for ($i = 0; $i -lt 120 -and -not $ok; $i++) {
            Start-Sleep -Seconds 1
            try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3; $ok = ($h.status -eq "ok") } catch {}
            if ($api.HasExited) { Fail "API process exited (code $($api.ExitCode))" }
        }
        if (-not $ok) { Fail "API did not become healthy on port $Port" }
        Write-Host ("Ready: {0} chunks indexed, LLM {1}" -f $h.index_points, $h.llm_model) -ForegroundColor Green
        try { Start-Process "http://127.0.0.1:$Port" } catch { Write-Host "Open http://127.0.0.1:$Port in your browser" }
        $api.WaitForExit()
    }
}
catch {
    Fail ($_ | Out-String)
}
finally {
    try { Stop-Transcript | Out-Null } catch {}
}
