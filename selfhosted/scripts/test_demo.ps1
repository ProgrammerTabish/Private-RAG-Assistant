<#
.SYNOPSIS
  Run the 13 HinSchG demo questions (eval\hinschg_demo_questions.xlsx) against the local demo index
  and write a scored report to data_ollama\reports (eval_*.xlsx / .json).
  ~1 min per question on a laptop CPU. Can run while the chat window is open.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\test_demo.ps1
  powershell -ExecutionPolicy Bypass -File selfhosted\scripts\test_demo.ps1 -Limit 3
#>
param([int]$Limit = 0, [string]$OllamaUrl = "http://127.0.0.1:11434")
& (Join-Path $PSScriptRoot "run_local_ollama.ps1") -Docs 24_HinSchG -SkipIndex -Limit $Limit `
    -EvalFile (Join-Path (Split-Path -Parent $PSScriptRoot) "eval\hinschg_demo_questions.xlsx") -OllamaUrl $OllamaUrl
