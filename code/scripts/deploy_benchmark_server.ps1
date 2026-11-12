[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_.@-]+$')]
    [string]$SshTarget,

    [Parameter(Mandatory = $true)]
    [string]$RemoteDirectory,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$RemotePython = "python3.11",

    [long]$MaxPriorDraws = 1000000000
)

$ErrorActionPreference = "Stop"

if ($RemoteDirectory -notmatch '^/[A-Za-z0-9._/-]+$' -or
    $RemoteDirectory -match '(^|/)\.\.(/|$)') {
    throw "RemoteDirectory must be an absolute Linux path without spaces or '..' segments."
}
if ($MaxPriorDraws -lt 1) {
    throw "MaxPriorDraws must be positive."
}

foreach ($toolName in @("ssh", "scp")) {
    if (-not (Get-Command $toolName -ErrorAction SilentlyContinue)) {
        throw "Required command '$toolName' was not found. Install/enable Windows OpenSSH Client."
    }
}

$codeRoot = Split-Path -Parent $PSScriptRoot
$requiredPaths = @(
    (Join-Path $codeRoot "src"),
    (Join-Path $codeRoot "configs"),
    (Join-Path $codeRoot "pyproject.toml"),
    (Join-Path $codeRoot "notebooks\benchmark_kaggle_standalone.py")
)
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required project file/directory is missing: $path"
    }
}

$pythonVersionOutput = & ssh $SshTarget "$RemotePython --version"
if ($LASTEXITCODE -ne 0) {
    throw "Could not run '$RemotePython --version' on $SshTarget."
}
if (($pythonVersionOutput -join " ") -notmatch 'Python\s+(\d+)\.(\d+)') {
    throw "Could not parse remote Python version: $pythonVersionOutput"
}
$remotePythonVersion = [version]::new(
    [int]$Matches[1],
    [int]$Matches[2],
    0
)
if ($remotePythonVersion -lt [version]"3.11") {
    throw "Remote Python 3.11+ is required for the selected PyTorch CUDA 12.8 wheel."
}

$gpuRows = @(
    & ssh $SshTarget "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0) {
    throw "nvidia-smi failed on $SshTarget; no accessible NVIDIA GPU was found."
}

$minimumDriver = [version]"570.26"
$a6000Found = $false
foreach ($gpuRow in $gpuRows) {
    $columns = "$gpuRow" -split ","
    if ($columns.Count -lt 3 -or $columns[0] -notmatch "RTX A6000") {
        continue
    }
    if ($columns[1].Trim() -notmatch '^(\d+(?:\.\d+){1,3})') {
        throw "Could not parse NVIDIA driver version from nvidia-smi row: $gpuRow"
    }
    $driverVersion = [version]$Matches[1]
    if ($columns[2].Trim() -notmatch '^(\d+)\s*MiB') {
        throw "Could not parse GPU memory from nvidia-smi row: $gpuRow"
    }
    $memoryMiB = [int]$Matches[1]
    if ($driverVersion -lt $minimumDriver) {
        throw "RTX A6000 driver $driverVersion is too old for the selected CUDA 12.8 runtime; Linux driver 570.26+ is required."
    }
    if ($memoryMiB -lt 45000) {
        throw "RTX A6000 reports only $memoryMiB MiB; expected a 48 GB card."
    }
    $a6000Found = $true
    Write-Host "Remote GPU: $($columns[0].Trim()), driver $driverVersion, $memoryMiB MiB"
}
if (-not $a6000Found) {
    throw "No RTX A6000 with at least 45000 MiB was found on $SshTarget. GPUs: $($gpuRows -join '; ')"
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "sfmpe-deploy-" + [guid]::NewGuid().ToString("N")
)
$packageRoot = Join-Path $tempRoot "package"
$archivePath = Join-Path $tempRoot "sfmpe-benchmark.zip"
$bootstrapPath = Join-Path $tempRoot "bootstrap.sh"

try {
    New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null
    foreach ($directoryName in @("src", "configs")) {
        $sourceDirectory = Join-Path $codeRoot $directoryName
        $destinationDirectory = Join-Path $packageRoot $directoryName
        New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
        Copy-Item -Path (Join-Path $sourceDirectory "*") `
            -Destination $destinationDirectory -Recurse -Force
    }

    New-Item -ItemType Directory -Path (Join-Path $packageRoot "notebooks") `
        -Force | Out-Null
    Copy-Item -LiteralPath `
        (Join-Path $codeRoot "notebooks\benchmark_kaggle_standalone.py") `
        -Destination (Join-Path $packageRoot "notebooks") -Force
    Copy-Item -LiteralPath (Join-Path $codeRoot "pyproject.toml") `
        -Destination $packageRoot -Force

    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    foreach ($configName in @(
        "sfmpe_benchmark_sir.yaml",
        "sfmpe_benchmark_stochvol.yaml"
    )) {
        $configPath = Join-Path (Join-Path $packageRoot "configs") $configName
        $configText = [System.IO.File]::ReadAllText($configPath)
        $pattern = '(?m)^([ \t]*max_prior_draws:[ \t]*)\d+([ \t]*(?:#.*)?)$'
        $matches = [regex]::Matches($configText, $pattern)
        if ($matches.Count -ne 1) {
            throw "Expected one max_prior_draws entry in bundled $configName; found $($matches.Count)."
        }
        $match = $matches[0]
        $replacement = $match.Groups[1].Value +
            $MaxPriorDraws.ToString() +
            $match.Groups[2].Value
        $configText = $configText.Substring(0, $match.Index) +
            $replacement +
            $configText.Substring($match.Index + $match.Length)
        [System.IO.File]::WriteAllText($configPath, $configText, $utf8NoBom)
    }

    Compress-Archive -Path (Join-Path $packageRoot "*") `
        -DestinationPath $archivePath -CompressionLevel Optimal

    $bootstrap = @'
#!/usr/bin/env bash
set -euo pipefail

RELEASE_DIR="${1:?release directory is required}"
REMOTE_PYTHON="${2:-python3.11}"
ARCHIVE="$RELEASE_DIR/sfmpe-benchmark.zip"
PROJECT="$RELEASE_DIR/project"
VENV="$PROJECT/.venv"

mkdir -p "$PROJECT"
"$REMOTE_PYTHON" -m zipfile -e "$ARCHIVE" "$PROJECT"
"$REMOTE_PYTHON" -m venv "$VENV"
PYTHON="$VENV/bin/python"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
"$PYTHON" -m pip install -e "$PROJECT"

"$PYTHON" - <<'PY'
import torch

if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected PyTorch CUDA 12.8, got {torch.version.cuda!r}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot initialize CUDA on this server")

matches = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    if (
        "RTX A6000" in properties.name
        and capability == (8, 6)
        and properties.total_memory >= 45_000_000_000
    ):
        matches.append((index, properties, capability))

if not matches:
    raise SystemExit(
        "CUDA is available, but no 48 GB RTX A6000 (compute capability 8.6) "
        "was detected by PyTorch"
    )

for index, properties, capability in matches:
    print(
        f"CUDA check passed: device={index}, name={properties.name}, "
        f"compute_capability={capability}, "
        f"memory_GiB={properties.total_memory / 1024**3:.1f}, "
        f"torch_cuda={torch.version.cuda}"
    )
PY

echo "Deployment ready at: $PROJECT"
echo "Run from that directory:"
echo "  .venv/bin/python notebooks/benchmark_kaggle_standalone.py --schedule configs/benchmark_schedule.yaml --shard sir-sfmpe --device cuda"
echo "  .venv/bin/python notebooks/benchmark_kaggle_standalone.py --schedule configs/benchmark_schedule.yaml --shard stochvol-sfmpe --device cuda"
'@
    [System.IO.File]::WriteAllText(
        $bootstrapPath,
        $bootstrap.Replace("`r`n", "`n"),
        $utf8NoBom
    )

    $releaseName = "sfmpe-benchmark-" + [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
    $remoteReleaseDirectory = $RemoteDirectory.TrimEnd("/") + "/" + $releaseName
    & ssh $SshTarget "mkdir -p $remoteReleaseDirectory"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create remote release directory $remoteReleaseDirectory."
    }

    $remoteDestination = "{0}:{1}/" -f $SshTarget, $remoteReleaseDirectory
    & scp $archivePath $bootstrapPath $remoteDestination
    if ($LASTEXITCODE -ne 0) {
        throw "Could not transfer the benchmark bundle to $remoteDestination."
    }

    $remoteBootstrapCommand = "bash $remoteReleaseDirectory/bootstrap.sh $remoteReleaseDirectory $RemotePython"
    & ssh $SshTarget $remoteBootstrapCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote installation or CUDA verification failed."
    }

    Write-Host "Deployment and CUDA verification succeeded."
    Write-Host "Remote project: $remoteReleaseDirectory/project"
    Write-Host "max_prior_draws in the bundled SIR and StochVol configs: $MaxPriorDraws"
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
