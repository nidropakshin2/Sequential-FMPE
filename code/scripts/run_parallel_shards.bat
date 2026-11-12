@echo off
setlocal
pushd "%~dp0\.."

@REM  set "KAGGLE_CONFIG_DIR=%USERPROFILE%\.kaggle"
@REM  start "gaussian-snpe-c" "%ComSpec%" /k "echo KAGGLE_CONFIG_DIR=%KAGGLE_CONFIG_DIR% && kagglex run --stream --dir . --file notebooks/benchmark_kaggle_standalone.py --include-data src pyproject.toml configs --gpu none --title gaussian-snpe-c --env SFMPE_BENCHMARK_CONFIG=benchmark.yaml SFMPE_BENCHMARK_SHARD=gaussian-snpe-c"
@REM  start "sir-snpe-c" "%ComSpec%" /k "echo KAGGLE_CONFIG_DIR=%KAGGLE_CONFIG_DIR% && kagglex run --stream --dir . --file notebooks/benchmark_kaggle_standalone.py --include-data src pyproject.toml configs --gpu none --title sir-snpe-c --env SFMPE_BENCHMARK_CONFIG=benchmark.yaml SFMPE_BENCHMARK_SHARD=sir-snpe-c"

set "KAGGLE_CONFIG_DIR=%USERPROFILE%\.kaggle-masha"
start "stochvol-sfmpe" "%ComSpec%" /k "echo KAGGLE_CONFIG_DIR=%KAGGLE_CONFIG_DIR% && kagglex run --stream --dir . --file notebooks/benchmark_kaggle_standalone.py --include-data src pyproject.toml configs --gpu none --title stochvol-sfmpe --env SFMPE_BENCHMARK_CONFIG=benchmark.yaml SFMPE_BENCHMARK_SHARD=stochvol-sfmpe"
start "sir-sfmpe" "%ComSpec%" /k "echo KAGGLE_CONFIG_DIR=%KAGGLE_CONFIG_DIR% && kagglex run --stream --dir . --file notebooks/benchmark_kaggle_standalone.py --include-data src pyproject.toml configs --gpu none --title sir-sfmpe --env SFMPE_BENCHMARK_CONFIG=benchmark.yaml SFMPE_BENCHMARK_SHARD=sir-sfmpe"

popd
endlocal
