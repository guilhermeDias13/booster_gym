<#
Envia o workspace de avaliação para uma máquina Vast e extrai em /app/eval_ws/.

  .\eval\upload.ps1 -HostIp 216.155.220.136 -Port 40822 -Label "avaliação, etapa 1"
  .\eval\upload.ps1 -HostIp 216.155.220.136 -Port 40822 -NoUpload   # só monta eval_ws.tar

Conteúdo do tar:
  eval/ (scripts, policies.yaml, EVAL_COMMIT, tests/)
  T1_sprint_arms_locked/code, T1_sprint_arms_not_locked/code
  envs/, utils/, resources/ da raiz (só para o T0_walk)
  de cada política com caminho: nn/model_10000.pth e config.yaml
#>
param(
    [Parameter(Mandatory = $true)][string]$HostIp,
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$Key = ".\key",
    [string]$Label = "avaliação",
    [switch]$NoUpload
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Host "# Máquina — ${HostIp}:${Port} — $Label"

# 1. As 3 árvores usadas pela avaliação precisam do patch de contato.
$t1 = @(
    "envs\t1.py",
    "T1_sprint_arms_locked\code\envs\t1.py",
    "T1_sprint_arms_not_locked\code\envs\t1.py"
)
$t1Hashes = [ordered]@{}
foreach ($path in $t1) {
    if (-not (Select-String -Path $path -Pattern "self.eval_mode = False" -Quiet)) {
        throw "$path não tem o patch eval_mode."
    }
    $t1Hashes[$path.Replace("\", "/")] = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}
Write-Host "Patch eval_mode presente nas 3 árvores."

# 2. EVAL_COMMIT: commit, indicador de dirty e sha256 dos scripts de avaliação.
$commit = (git rev-parse HEAD).Trim()
$dirty = [bool](git status --porcelain -- eval envs/t1.py)
$pySha = [ordered]@{}
Get-ChildItem eval -Recurse -Include *.py | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($repo.Length + 1).Replace("\", "/")
    $pySha[$rel] = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower()
}
$info = [ordered]@{
    commit          = $commit
    dirty           = $dirty
    written_at      = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")
    t1_py_sha256    = $t1Hashes
    eval_py_sha256  = $pySha
}
$info | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 eval\EVAL_COMMIT
Write-Host "EVAL_COMMIT: $commit (dirty=$dirty)"

# 3. Lista de arquivos.
$files = New-Object System.Collections.Generic.List[string]
function Add-Tree([string]$dir) {
    Get-ChildItem $dir -Recurse -File |
        Where-Object { $_.FullName -notmatch "\\__pycache__\\" -and $_.Extension -ne ".pyc" } |
        ForEach-Object { $files.Add($_.FullName.Substring($repo.Length + 1).Replace("\", "/")) }
}
Get-ChildItem eval -File | Where-Object { $_.Extension -in ".py", ".yaml", ".md" -or $_.Name -eq "EVAL_COMMIT" } |
    ForEach-Object { $files.Add("eval/" + $_.Name) }
if (Test-Path eval\tests) { Add-Tree "eval\tests" }
foreach ($d in "T1_sprint_arms_locked\code", "T1_sprint_arms_not_locked\code", "envs", "utils", "resources") {
    Add-Tree $d
}

$policies = py -c "import sys; sys.path.insert(0, 'eval'); import common as c; [print(e['name'] + '|' + e['checkpoint']) for e in c.load_policies()]"
if ($LASTEXITCODE -ne 0) { throw "Falha ao ler eval/policies.yaml (precisa de 'py' com pyyaml)." }
$missing = @()
foreach ($line in $policies) {
    $name, $ck = $line.Split("|", 2)
    if ($ck -like "TODO*") { Write-Warning "$name sem checkpoint (TODO), não enviado"; continue }
    $ckLocal = $ck.Replace("/", "\")
    $cfg = Join-Path (Split-Path -Parent (Split-Path -Parent $ckLocal)) "config.yaml"
    foreach ($f in $ckLocal, $cfg) {
        if (Test-Path $f) { $files.Add($f.Replace("\", "/")) } else { $missing += "${name}: $f" }
    }
}
if ($missing.Count -gt 0) { throw ("Arquivos em falta:`n  " + ($missing -join "`n  ")) }

$list = Join-Path $env:TEMP "eval_ws_files.txt"
[System.IO.File]::WriteAllLines($list, ($files | Sort-Object -Unique))
$tarPath = Join-Path $env:TEMP "eval_ws.tar"
if (Test-Path $tarPath) { Remove-Item $tarPath }
tar -cf $tarPath -T $list
if ($LASTEXITCODE -ne 0) { throw "tar falhou" }
Write-Host ("eval_ws.tar: {0} arquivos, {1:N1} MB" -f ($files | Sort-Object -Unique).Count, ((Get-Item $tarPath).Length / 1MB))

if ($NoUpload) { return }

# 4. Envio e extração.
# Máquina — ${HostIp}:${Port} — $Label
scp -P $Port -i $Key -o StrictHostKeyChecking=accept-new $tarPath "root@${HostIp}:/app/eval_ws.tar"
if ($LASTEXITCODE -ne 0) { throw "scp falhou" }
ssh -p $Port -i $Key -o StrictHostKeyChecking=accept-new "root@$HostIp" "mkdir -p /app/eval_ws && tar -xf /app/eval_ws.tar -C /app/eval_ws && rm /app/eval_ws.tar && cd /app/eval_ws && sha256sum envs/t1.py */code/envs/t1.py && (python3 -c 'import scipy' 2>/dev/null || pip install -q scipy) && echo OK"
if ($LASTEXITCODE -ne 0) { throw "extração no Vast falhou" }
Write-Host "Pronto: /app/eval_ws em ${HostIp}:${Port}"
