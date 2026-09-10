param(
    [string]$Conda = '',
    [string]$Environment = 'py314',
    [ValidateSet('collect', 'smoke', 'serial', 'parallel', 'docs')]
    [string]$Check = 'serial'
)
$ErrorActionPreference = 'Stop'
if (-not $Conda) {
    $Conda = @('C:\ProgramData\miniforge3\Scripts\conda.exe',
               'C:\ProgramData\Anaconda3\Scripts\conda.exe') |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Conda) { throw 'No supported Conda installation found; supply -Conda.' }
$repo = Split-Path $PSScriptRoot -Parent
$output = Join-Path $PSScriptRoot ('runs/' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + $Check + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $output -Force | Out-Null
$names = @('MPLBACKEND', 'QT_QPA_PLATFORM', 'READTHEDOCS', 'COVERAGE_FILE')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
Push-Location $repo
try {
    $env:MPLBACKEND = 'Agg'
    $env:QT_QPA_PLATFORM = 'offscreen'
    $env:READTHEDOCS = 'False'
    $env:COVERAGE_FILE = Join-Path $output '.coverage'
    $arguments = @('-m', 'pytest', '-vv', ('--basetemp=' + (Join-Path $output 'tmp')), '-o', 'faulthandler_timeout=120')
    switch ($Check) {
        'collect' { $arguments += '--collect-only' }
        'smoke' {
            $arguments += @('tests/stoner/test_Core.py', 'tests/stoner/test_FileFormats.py',
                'tests/stoner/plot/test_plot.py', 'tests/stoner/folders/test_Folders.py',
                'tests/stoner/image/test_core.py')
        }
        'parallel' { $arguments += @('-n', '2') }
        'docs' {
            $env:READTHEDOCS = 'True'
            $arguments = @('-m', 'sphinx', '-b', 'html', '-E', 'doc',
                (Join-Path $repo 'doc/_build/baseline'), '-w', (Join-Path $output 'sphinx-warnings.log'))
        }
    }
    if ($Check -in @('serial', 'parallel')) {
        $arguments += @('--cov=Stoner', '--cov-report=term', ('--cov-report=xml:' + (Join-Path $output 'coverage.xml')))
    }
    if ($Check -ne 'docs') { $arguments += '--junitxml=' + (Join-Path $output 'pytest.xml') }
    $started = Get-Date
    & $Conda run --no-capture-output -n $Environment python @arguments *> (Join-Path $output 'run.log')
    $result = $LASTEXITCODE
    [ordered]@{check=$Check; environment=$Environment; conda=$Conda; arguments=$arguments;
        started=$started.ToString('o'); elapsedSeconds=((Get-Date)-$started).TotalSeconds;
        exitCode=$result} | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $output 'result.json')
    Get-Content (Join-Path $output 'run.log') -Tail 20
    Write-Output "Reports: $output"
}
finally {
    Pop-Location
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
exit $result