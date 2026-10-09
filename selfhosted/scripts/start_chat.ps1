<#
.SYNOPSIS
  Start the local self-hosted RAG chat (Ollama BGE-M3 + 3B LLM, Qdrant server, FastAPI + UI)
  and open it in the browser.
  Default = small DEMO with one law (24_HinSchG.pdf, 68 chunks, ~3-5 min first start on a laptop CPU).
  -AllDocs = all 50 documents (keyword search for all + 5 min BGE-M3 budget per run, -Improve adds more).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\start_chat.ps1
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\start_chat.ps1 -Docs 01_GwG    (bigger: 318 chunks, ~12 min)
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\start_chat.ps1 -AllDocs -Improve
#>
param([string]$LlmModel = "auto", [int]$Port = 8000, [string[]]$Docs = @("24_HinSchG"), [switch]$AllDocs,
      [int]$EmbedMinutes = 5, [switch]$Improve, [switch]$Full,
      [string]$OllamaUrl = "http://127.0.0.1:11434")
if ($AllDocs) { $Docs = @() }
& (Join-Path $PSScriptRoot "run_local_ollama.ps1") -NoEval -Serve -LlmModel $LlmModel -Port $Port `
    -Docs $Docs -EmbedMinutes $EmbedMinutes -Improve:$Improve -Full:$Full -OllamaUrl $OllamaUrl
