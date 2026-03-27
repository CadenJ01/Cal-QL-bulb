$ErrorActionPreference = "Stop"

$pairs = @(
  @("JaxCQL\\checkpointing.py", "/home/caden/calql-wsl/JaxCQL/checkpointing.py"),
  @("JaxCQL\\manifeel_sampler.py", "/home/caden/calql-wsl/JaxCQL/manifeel_sampler.py"),
  @("JaxCQL\\conservative_sac.py", "/home/caden/calql-wsl/JaxCQL/conservative_sac.py"),
  @("JaxCQL\\conservative_sac_main.py", "/home/caden/calql-wsl/JaxCQL/conservative_sac_main.py"),
  @("scripts\\run_bulb_offline_online_wsl.sh", "/home/caden/calql-wsl/scripts/run_bulb_offline_online_wsl.sh")
)

foreach ($pair in $pairs) {
  $src = "C:\Users\26972\Desktop\Spring 2026\cs441\calql\" + $pair[0]
  $dst = $pair[1]
  $dstDir = $dst.Substring(0, $dst.LastIndexOf('/'))
  $wslSrc = "/mnt/c/Users/26972/Desktop/Spring 2026/cs441/calql/" + $pair[0].Replace('\','/')
  Write-Host "Syncing $src -> $dst"
  wsl -d Ubuntu bash -lc "mkdir -p '$dstDir' && cp '$wslSrc' '$dst'"
}
