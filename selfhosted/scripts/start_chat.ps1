<#
.SYNOPSIS
  Start the local self-hosted RAG chat (Ollama BGE-M3 + 3B LLM, Qdrant server, FastAPI + UI)
  and open it in the browser. Builds the index on the first run; later runs reuse it.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\start_chat.ps1
#>
param([string]$LlmModel = "auto", [int]$Port = 8000)
& (Join-Path $PSScriptRoot "run_local_ollama.ps1") -NoEval -Serve -LlmModel $LlmModel -Port $Port
