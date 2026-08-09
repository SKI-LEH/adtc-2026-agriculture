# Baseline sweep: run the official ADTC profiler (--skip-accuracy) on each
# candidate GGUF, then a direct llama-bench thread-scaling probe.
# Emits per-model submission.json + a summary table with S_perf / S_eff estimates.
# ASCII-only on purpose (Windows PowerShell 5.1 reads no-BOM files as CP1252).
#
# Scoring reference (from adtc-profiler source):
#   S_perf = min(TPS / 15, 1) * 100
#   S_eff  = max(0, (7 - peak_rss_gb) / 7) * 100
#   Hard cap: peak RSS must stay under the 7.5 GB audit Docker limit.

$ErrorActionPreference = "Stop"
$llamaDir = "C:\Users\starl\source\repos\hackathon\tools\llama"
$py       = "C:\Users\starl\source\repos\hackathon\.venv\Scripts\python.exe"
$root     = "C:\Users\starl\source\repos\hackathon\adtc-agriculture\benchmarks\candidates"
$env:PATH = "$llamaDir;$env:PATH"
$env:PYTHONPATH = "C:\Users\starl\source\repos\hackathon\adtc-profiler\src"
# Profiler prints unicode (arrows); this old console is cp1252. Force Python UTF-8
# so rich can encode its output instead of crashing on UnicodeEncodeError.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$cands = @("qwen2.5-0.5b","llama3.2-1b","qwen2.5-1.5b")
$rows = @()

foreach ($c in $cands) {
  $dir = "$root\$c"
  $meta = Get-Content "$dir\metadata.json" -Raw | ConvertFrom-Json
  $model = "$dir\" + ($meta._runtime.model_path -replace '/','\')
  if (-not (Test-Path $model)) { Write-Host "MISSING model for $c - skipping"; continue }

  Write-Host "=== profiling $c ==="
  & $py -m adtc_profiler.cli run --submission $dir --mode participant --skip-accuracy --output "$dir\submission.json"
  if ($LASTEXITCODE -ne 0) { Write-Host "profiler FAILED for $c (exit $LASTEXITCODE)"; continue }

  $r = Get-Content "$dir\submission.json" -Raw | ConvertFrom-Json
  $tps      = [double]$r.throughput.tokens_per_second_generation
  $ttft     = [double]$r.throughput.first_token_latency_ms
  $peakGb   = [double]$r.memory.peak_rss_mb / 1024.0
  $sPerf    = [Math]::Min($tps / 15.0, 1.0) * 100.0
  $sEff     = [Math]::Max(0.0, (7.0 - $peakGb) / 7.0) * 100.0

  $rows += [pscustomobject]@{
    model      = $c
    TPS_2core  = [Math]::Round($tps,2)
    S_perf     = [Math]::Round($sPerf,1)
    peak_GB    = [Math]::Round($peakGb,2)
    S_eff      = [Math]::Round($sEff,1)
    TTFT_ms    = [Math]::Round($ttft,0)
    params     = $r.model_info.params_count
    params_ok  = $r.model_info.params_match
  }
}

# Direct llama-bench thread-scaling probe (this box has 2 cores; audit VM has 4 vCPU).
Write-Host "=== thread-scaling probe on 1.5B (1 then 2 threads) ==="
$probeModel = "$root\qwen2.5-1.5b\model\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
if (Test-Path $probeModel) {
  foreach ($t in @(1,2)) {
    try {
      # llama-bench writes backend-load lines to stderr; under ErrorActionPreference=Stop
      # PS 5.1 turns native stderr into a terminating error, so drop to Continue for the call.
      $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
      $j = & "$llamaDir\llama-bench.exe" -m $probeModel -p 512 -n 128 -ngl 0 -t $t --output json 2>$null | Out-String
      $ErrorActionPreference = $eap
      $arr = $j | ConvertFrom-Json
      $tg = $arr | Where-Object { $_.n_gen -gt 0 } | Select-Object -First 1
      "  1.5B @ {0} thread(s): {1:N2} tok/s" -f $t, ([double]$tg.avg_ts)
    } catch { "  probe t=$t failed: $($_.Exception.Message)" }
  }
}

Write-Host "=== SWEEP SUMMARY (TPS is 2-core; extrapolate to 4 vCPU) ==="
$rows | Format-Table -AutoSize | Out-String | Write-Host
$rows | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 "C:\Users\starl\source\repos\hackathon\adtc-agriculture\benchmarks\sweep_results.json"
Write-Host "wrote benchmarks\sweep_results.json"