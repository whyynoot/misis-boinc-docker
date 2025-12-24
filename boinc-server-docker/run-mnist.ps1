param(
    [string]$WuName = "mnist_run$(Get-Date -Format 'yyyyMMddHHmmss')",
    [int]$PollSeconds = 240,
    [int]$PollIntervalSeconds = 5,
    [int]$Rounds = 3,
    [int]$MinQuorum = 2,
    [int]$TargetResults = 2,
    [int]$SampleCount = 2000,
    [int]$TrainEpochs = 5,
    [int]$BatchSize = 256,
    [double]$LearningRate = 0.1
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$null = docker compose exec apache chown -R boincadm:boincadm /results

function Get-ResultState {
    param([string]$ResultName)
    $sql = "SELECT r.server_state, r.outcome, r.validate_state, w.assimilate_state, w.file_delete_state FROM result r JOIN workunit w ON w.id = r.workunitid WHERE r.name='${ResultName}';"
    $oldErrPref = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $raw = docker compose exec -T mysql mysql -uroot boincserver -N -e "$sql" 2>&1
    } finally {
        $ErrorActionPreference = $oldErrPref
    }
    if ($LASTEXITCODE -ne 0) {
        throw "MySQL query failed: $raw"
    }
    $line = $raw | Where-Object { $_ -match "^\s*\d" } | Select-Object -First 1
    if (-not $line) {
        return $null
    }
    $parts = $line -split "\s+"
    if ($parts.Count -lt 5) {
        return $null
    }
    return [pscustomobject]@{
        ServerState     = [int]$parts[0]
        Outcome         = [int]$parts[1]
        ValidateState   = [int]$parts[2]
        AssimilateState = [int]$parts[3]
        FileDeleteState = [int]$parts[4]
    }
}

function Get-ClientServices {
    docker compose ps --services | Where-Object { $_ -match "^client\d+$" }
}

function Update-Clients {
    param([string]$ProjectUrl)
    foreach ($client in (Get-ClientServices)) {
        docker compose exec $client boinccmd --project $ProjectUrl update | Out-Null
    }
}

for ($r = 1; $r -le $Rounds; $r++) {
    $wuThis = "${WuName}_r${r}"
    Write-Host "`nSubmitting MNIST workunit $wuThis (round $r/$Rounds)..."

    $payload = @{
        sample_count  = $SampleCount
        seed          = [int](Get-Random -Maximum 1000000)
        round         = $r
        train_epochs  = $TrainEpochs
        batch_size    = $BatchSize
        learning_rate = $LearningRate
    } | ConvertTo-Json -Compress

    $jobFileName = "${wuThis}.json"
    $jobPath = "/home/boincadm/project/download/$jobFileName"
    $payload | docker compose exec -T apache bash -lc "cat > $jobPath"

    $createCmd = @"
cd /home/boincadm/project && bin/create_work --appname mnist --wu_name $wuThis --wu_template templates/mnist_in --result_template templates/mnist_out --min_quorum $MinQuorum --target_nresults $TargetResults --rsc_fpops_est 1e12 --rsc_fpops_bound 1e13 --rsc_memory_bound 2000000000 --rsc_disk_bound 500000000 $jobFileName
"@
    docker compose exec apache bash -lc $createCmd

    # Nudge all clients to fetch work immediately
    Update-Clients -ProjectUrl "http://host.docker.internal:8082/boincserver"

    # Wait for the daemons (scheduler/transitioner/validator/assimilator/file_deleter) to process the job
    $deadline = (Get-Date).AddSeconds($PollSeconds)
    $resultName = "${wuThis}_0"
    $state = $null

    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds $PollIntervalSeconds
        Update-Clients -ProjectUrl "http://host.docker.internal:8082/boincserver"
        $state = Get-ResultState -ResultName $resultName
        if ($state) {
            Write-Host ("State server={0} outcome={1} validate={2} assimilate={3} file_delete={4}" -f $state.ServerState, $state.Outcome, $state.ValidateState, $state.AssimilateState, $state.FileDeleteState)
        }
        if ($state -and $state.ServerState -eq 5 -and $state.ValidateState -eq 1 -and $state.AssimilateState -eq 2) {
            break
        }
    }

    if (-not $state -or $state.ValidateState -ne 1 -or $state.AssimilateState -ne 2) {
        $stateMsg = if ($state) { "server=$($state.ServerState) outcome=$($state.Outcome) validate=$($state.ValidateState) assimilate=$($state.AssimilateState) file_delete=$($state.FileDeleteState)" } else { "no state yet" }
        Write-Warning "Workunit $wuThis did not validate/assimilate within $PollSeconds seconds (last state: $stateMsg)."
        Pop-Location
        exit 1
    }

    $listCmd = "ls -1 /results/mnist/$resultName 2>/dev/null"
    $files = docker compose exec apache bash -lc $listCmd
    Write-Host "Workunit $wuThis validated and assimilated by BOINC daemons."
    if ($files) {
        Write-Host "Result artifacts under /results/mnist/$resultName inside the apache container:"
        $files | ForEach-Object { Write-Host " - $_" }
    } else {
        Write-Host "No files found under /results/mnist/$resultName; check assimilator logs if needed."
    }
}

Pop-Location
