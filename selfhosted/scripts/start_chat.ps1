<#
.SYNOPSIS
  Start the local self-hosted RAG chat (Ollama BGE-M3 + 3B LLM, Qdrant server, FastAPI + UI)
  and open it in the browser.
  First start: about 10 minutes (keyword index for all chunks + 5 min of BGE-M3 vectors).
  Later starts: about 1-2 minutes. Add -Improve to spend another 5 min on BGE-M3 coverage.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\start_chat.ps1
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\start_chat.ps1 -Improve
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\start_chat.ps1 -Improve -EmbedMinutes 30
#>
param([string]$LlmModel = "auto", [int]$Port = 8000, [int]$EmbedMinutes = 5, [switch]$Improve, [switch]$Full)
& (Join-Path $PSScriptRoot "run_local_ollama.ps1") -NoEval -Serve -LlmModel $LlmModel -Port $Port `
    -EmbedMinutes $EmbedMinutes -Improve:$Improve -Full:$Full
