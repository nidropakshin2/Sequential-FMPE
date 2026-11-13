[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_.@-]+$')]
    [string]$SshTarget,

    [Parameter(Mandatory = $true)]
    [string]$RemoteDirectory,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$RemotePython = "python3.11",

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$Shard = "all-tasks-sfmpe",

    [ValidateSet("RTXA6000", "RTXA4000")]
    [string]$GpuProfile = "RTXA6000"
)

$ErrorActionPreference = "Stop"

if ($RemoteDirectory -notmatch '^/[A-Za-z0-9._/-]+$' -or
    $RemoteDirectory -match '(^|/)\.\.(/|$)') {
    throw "RemoteDirectory must be an absolute Linux path without spaces or '..' segments."
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
if ($remotePythonVersion -lt [version]"3.10") {
    throw "Remote Python 3.10+ is required for the selected PyTorch CUDA 12.8 wheel."
}

$gpuRequirements = @{
    RTXA6000 = @{
        Name = "RTX A6000"
        MinimumMemoryMiB = 45000
    }
    RTXA4000 = @{
        Name = "RTX A4000"
        MinimumMemoryMiB = 14000
    }
}
$gpuRequirement = $gpuRequirements[$GpuProfile]
$gpuRows = @(
    & ssh $SshTarget "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
)
if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -eq 0) {
    throw "nvidia-smi failed on $SshTarget; no accessible NVIDIA GPU was found."
}

$minimumDriver = [version]"570.26"
$compatibleGpuFound = $false
foreach ($gpuRow in $gpuRows) {
    $columns = "$gpuRow" -split ","
    if ($columns.Count -lt 3 -or $columns[0] -notmatch [regex]::Escape($gpuRequirement.Name)) {
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
    if ($memoryMiB -lt $gpuRequirement.MinimumMemoryMiB) {
        throw "$($gpuRequirement.Name) reports only $memoryMiB MiB; expected at least $($gpuRequirement.MinimumMemoryMiB) MiB."
    }
    $compatibleGpuFound = $true
    Write-Host "Remote GPU: $($columns[0].Trim()), driver $driverVersion, $memoryMiB MiB"
}
if (-not $compatibleGpuFound) {
    throw "No $($gpuRequirement.Name) with at least $($gpuRequirement.MinimumMemoryMiB) MiB was found on $SshTarget. GPUs: $($gpuRows -join '; ')"
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

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::Open(
        $archivePath,
        [System.IO.Compression.ZipArchiveMode]::Create
    )
    try {
        foreach ($file in Get-ChildItem -LiteralPath $packageRoot -File -Recurse) {
            if ($file.FullName -match '[\\/](?:__pycache__|\.venv)[\\/]') {
                continue
            }
            if ($file.Extension -eq ".pyc") {
                continue
            }
            $entryName = $file.FullName.Substring($packageRoot.Length).
                TrimStart([char[]]@('\', '/')) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $file.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $archive.Dispose()
    }

    $bootstrap = @'
#!/usr/bin/env bash
set -euo pipefail

RELEASE_DIR="${1:?release directory is required}"
REMOTE_PYTHON="${2:-python3.11}"
SHARD="${3:?benchmark shard is required}"
GPU_PROFILE="${4:?GPU profile is required}"
ARCHIVE="$RELEASE_DIR/sfmpe-benchmark.zip"
PROJECT="$RELEASE_DIR/project"
VENV="$PROJECT/.venv"

mkdir -p "$PROJECT"
"$REMOTE_PYTHON" -m zipfile -e "$ARCHIVE" "$PROJECT"
"$REMOTE_PYTHON" -m venv "$VENV"
PYTHON="$VENV/bin/python"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install numpy torch

GPU_PROFILE="$GPU_PROFILE" "$PYTHON" - <<'PY'
import os

import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot initialize CUDA on this server")

matches = []
gpu_profiles = {
    "RTXA6000": ("RTX A6000", 8, 6, 45_000_000_000),
    "RTXA4000": ("RTX A4000", 8, 6, 14_000_000_000),
}
profile = os.environ["GPU_PROFILE"]
try:
    expected_name, major, minor, minimum_memory = gpu_profiles[profile]
except KeyError:
    raise SystemExit(f"Unsupported GPU profile: {profile!r}") from None

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    if (
        expected_name in properties.name
        and capability == (major, minor)
        and properties.total_memory >= minimum_memory
    ):
        matches.append((index, properties, capability))

if not matches:
    raise SystemExit(
        f"CUDA is available, but no {expected_name} with compute capability "
        f"{(major, minor)} and at least {minimum_memory} bytes was detected by PyTorch"
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
cd "$PROJECT"
LOG_FILE="$RELEASE_DIR/$SHARD.log"
PID_FILE="$RELEASE_DIR/$SHARD.pid"
STATUS_FILE="$RELEASE_DIR/$SHARD.exit-code"
rm -f "$PID_FILE" "$STATUS_FILE"
STATUS_FILE="$STATUS_FILE" nohup bash -c '
set +e
"$@"
status=$?
printf "%s\n" "$status" > "$STATUS_FILE"
exit "$status"
' bash "$PYTHON" notebooks/benchmark_kaggle_standalone.py \
    --schedule configs/benchmark_schedule.yaml \
    --shard "$SHARD" \
    --device cuda > "$LOG_FILE" 2>&1 < /dev/null &
BENCHMARK_PID=$!
printf "%s\n" "$BENCHMARK_PID" > "$PID_FILE"
sleep 5
if [ -f "$STATUS_FILE" ]; then
    STATUS="$(cat "$STATUS_FILE")"
    if [ "$STATUS" -ne 0 ]; then
        cat "$LOG_FILE"
        exit "$STATUS"
    fi
    echo "Benchmark shard $SHARD completed successfully."
elif kill -0 "$BENCHMARK_PID" 2>/dev/null; then
    echo "Benchmark shard $SHARD is running with PID $BENCHMARK_PID."
else
    cat "$LOG_FILE"
    exit 1
fi
echo "Benchmark log: $LOG_FILE"
echo "Benchmark exit status file: $STATUS_FILE"
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

    $remoteBootstrapCommand = "bash $remoteReleaseDirectory/bootstrap.sh $remoteReleaseDirectory $RemotePython $Shard $GpuProfile"
    & ssh $SshTarget $remoteBootstrapCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote installation, CUDA verification, or shard '$Shard' failed."
    }

    Write-Host "Deployment and CUDA verification succeeded; shard '$Shard' was launched."
    Write-Host "Remote project: $remoteReleaseDirectory/project"
    Write-Host "Benchmark shard: $Shard"
    Write-Host "GPU profile: $GpuProfile"
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
