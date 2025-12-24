param(
    [string]$WuName = "cifar100_run$(Get-Date -Format 'yyyyMMddHHmmss')",
    [int]$PollSeconds = 1800,
    [int]$PollIntervalSeconds = 10,
    [int]$Rounds = 3,
    [int]$MinQuorum = 2,
    [int]$TargetResults = 10,
    [bool]$PerHost = $false,
    [int]$ClientCount = 10,
    [int]$SampleCount = 50000,
    [int]$ValCount = 10000,
    [int]$TrainEpochs = 10,
    [int]$BatchSize = 128,
    [double]$LearningRate = 0.0005,
    [double]$WeightDecay = 0.05,
    [double]$LabelSmoothing = 0.0,
    [int]$InputSize = 32,
    [bool]$UsePretrained = $true,
    [bool]$FreezeBackbone = $false,
    [int]$NumWorkers = 2,
    [int]$NumThreads = 2,
    [bool]$ForceCpu = $false,
    [bool]$UseClientHash = $true,
    [bool]$ComputeConfusion = $true
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

$null = docker compose exec apache chown -R boincadm:boincadm /results

function Ensure-AssignmentsEnabled {
    $cmd = "grep -c '<enable_assignment>1</enable_assignment>' /home/boincadm/project/config.xml || true"
    $countRaw = docker compose exec -T apache bash -lc $cmd
    $count = 0
    [int]::TryParse($countRaw, [ref]$count) | Out-Null
    if ($count -lt 1) {
        throw "Per-host submission requires <enable_assignment>1</enable_assignment> in config.xml inside the apache container."
    }
}

function Invoke-CreateWork {
    param(
        [string]$WuName,
        [string]$Command
    )
    Write-Host "create_work for $WuName ..."
    $output = docker compose exec apache bash -lc $Command 2>&1
    $exit = $LASTEXITCODE
    foreach ($line in $output) {
        if ($line) { Write-Host "  $line" }
    }
    if ($exit -ne 0) {
        throw "create_work failed for $WuName (exit $exit)"
    }
}

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

function Get-ClientHosts {
    $sql = "SELECT id, domain_name FROM host WHERE domain_name LIKE 'client%' ORDER BY domain_name;"
    $raw = docker compose exec -T mysql mysql -uroot boincserver -N -e "$sql" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "MySQL query failed: $raw"
    }
    $hosts = @()
    foreach ($line in $raw) {
        if (-not $line) { continue }
        $parts = $line -split "\s+"
        if ($parts.Count -lt 2) { continue }
        $hosts += [pscustomobject]@{
            Id = [int]$parts[0]
            Name = $parts[1].Trim()
        }
    }
    return $hosts
}

function Get-ResultNamesForWu {
    param([string]$WuName)
    $sql = "SELECT r.name FROM result r JOIN workunit w ON w.id = r.workunitid WHERE w.name='${WuName}';"
    $raw = docker compose exec -T mysql mysql -uroot boincserver -N -e "$sql" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "MySQL query failed: $raw"
    }
    $names = @()
    foreach ($line in $raw) {
        if (-not $line) { continue }
        $names += $line.Trim()
    }
    return $names
}

function Get-AssignmentsForWu {
    param([string]$WuName)
    $sql = "SELECT a.id, a.target_id, a.multi FROM assignment a JOIN workunit w ON a.workunitid=w.id WHERE w.name='${WuName}';"
    $raw = docker compose exec -T mysql mysql -uroot boincserver -N -e "$sql" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "MySQL query failed: $raw"
    }
    $assignments = @()
    foreach ($line in $raw) {
        if (-not $line) { continue }
        $parts = $line -split "\s+"
        if ($parts.Count -lt 3) { continue }
        $assignments += [pscustomobject]@{
            Id       = [int]$parts[0]
            TargetId = [int]$parts[1]
            Multi    = [int]$parts[2]
        }
    }
    return $assignments
}

function Update-Clients {
    param([string]$ProjectUrl)
    foreach ($client in (Get-ClientServices)) {
        docker compose exec $client boinccmd --project $ProjectUrl update | Out-Null
    }
}

$roundSummaries = @()

$clientHosts = @()
if ($PerHost) {
    Ensure-AssignmentsEnabled
    $clientHosts = Get-ClientHosts
    if (-not $clientHosts -or $clientHosts.Count -eq 0) {
        throw "No client hosts registered in DB. Make sure clients are attached and scheduler is healthy."
    }
    if ($ClientCount -gt 0) {
        if ($clientHosts.Count -lt $ClientCount) {
            throw "Requested $ClientCount clients, but only $($clientHosts.Count) hosts are registered."
        }
        $clientHosts = $clientHosts | Select-Object -First $ClientCount
    }
    Write-Host ("Using {0} hosts: {1}" -f $clientHosts.Count, (($clientHosts | ForEach-Object { $_.Name }) -join ", "))
}

for ($r = 1; $r -le $Rounds; $r++) {
    $roundSeed = [int](Get-Random -Maximum 1000000)
    $wuNames = @()
    if ($PerHost) {
        Write-Host "`nSubmitting CIFAR-100 workunits for round $r/$Rounds..."
        foreach ($clientHost in $clientHosts) {
            $hostLabel = $clientHost.Name
            if (-not $hostLabel) {
                $hostLabel = "host$($clientHost.Id)"
            }
            $hostLabel = $hostLabel -replace "[^A-Za-z0-9_.-]", "_"
            $wuThis = "${WuName}_r${r}_$hostLabel"
            $wuNames += $wuThis
            $payload = @{
                sample_count      = $SampleCount
                val_count         = $ValCount
                seed              = $roundSeed
                round             = $r
                client_label      = $clientHost.Name
                train_epochs      = $TrainEpochs
                batch_size        = $BatchSize
                learning_rate     = $LearningRate
                weight_decay      = $WeightDecay
                label_smoothing   = $LabelSmoothing
                input_size        = $InputSize
                use_pretrained    = [bool]$UsePretrained
                freeze_backbone   = [bool]$FreezeBackbone
                force_cpu         = [bool]$ForceCpu
                num_workers       = $NumWorkers
                num_threads       = $NumThreads
                use_client_hash   = [bool]$UseClientHash
                compute_confusion = [bool]$ComputeConfusion
            } | ConvertTo-Json -Compress

            $jobFileName = "${wuThis}.json"
            $jobPath = "/home/boincadm/project/download/$jobFileName"
            $payload | docker compose exec -T apache bash -lc "cat > $jobPath"

            $createCmd = @"
cd /home/boincadm/project && bin/create_work --appname cifar100 --app_version_num 100 --wu_name $wuThis --wu_template templates/cifar100_in --result_template templates/cifar100_out --min_quorum 1 --target_nresults 1 --target_host $($clientHost.Id) --rsc_fpops_est 2e12 --rsc_fpops_bound 5e13 --rsc_memory_bound 4000000000 --rsc_disk_bound 2000000000 $jobFileName
"@
            Invoke-CreateWork -WuName $wuThis -Command $createCmd
            $assignments = Get-AssignmentsForWu -WuName $wuThis
            if (-not $assignments -or $assignments.Count -eq 0) {
                Write-Warning "No assignments recorded for $wuThis; scheduler may ignore it."
            } else {
                $targets = $assignments | ForEach-Object { $_.TargetId } | Sort-Object -Unique
                Write-Host ("Assignments for {0}: {1}" -f $wuThis, ($targets -join ","))
            }
        }
    } else {
        $wuThis = "${WuName}_r${r}"
        $wuNames += $wuThis
        Write-Host "`nSubmitting CIFAR-100 workunit $wuThis (round $r/$Rounds)..."
        $payload = @{
            sample_count      = $SampleCount
            val_count         = $ValCount
            seed              = $roundSeed
            round             = $r
            train_epochs      = $TrainEpochs
            batch_size        = $BatchSize
            learning_rate     = $LearningRate
            weight_decay      = $WeightDecay
            label_smoothing   = $LabelSmoothing
            input_size        = $InputSize
            use_pretrained    = [bool]$UsePretrained
            freeze_backbone   = [bool]$FreezeBackbone
            force_cpu         = [bool]$ForceCpu
            num_workers       = $NumWorkers
            num_threads       = $NumThreads
            use_client_hash   = [bool]$UseClientHash
            compute_confusion = [bool]$ComputeConfusion
        } | ConvertTo-Json -Compress

        $jobFileName = "${wuThis}.json"
        $jobPath = "/home/boincadm/project/download/$jobFileName"
        $payload | docker compose exec -T apache bash -lc "cat > $jobPath"

        $createCmd = @"
cd /home/boincadm/project && bin/create_work --appname cifar100 --app_version_num 100 --wu_name $wuThis --wu_template templates/cifar100_in --result_template templates/cifar100_out --min_quorum $MinQuorum --target_nresults $TargetResults --rsc_fpops_est 2e12 --rsc_fpops_bound 5e13 --rsc_memory_bound 4000000000 --rsc_disk_bound 2000000000 $jobFileName
"@
        Invoke-CreateWork -WuName $wuThis -Command $createCmd
    }

    Update-Clients -ProjectUrl "http://host.docker.internal:8082/boincserver"

    $deadline = (Get-Date).AddSeconds($PollSeconds)
    $pending = @{}
    $roundStart = Get-Date
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds $PollIntervalSeconds
        Update-Clients -ProjectUrl "http://host.docker.internal:8082/boincserver"
        $elapsed = [int]((Get-Date) - $roundStart).TotalSeconds
        foreach ($wu in $wuNames) {
            foreach ($rname in (Get-ResultNamesForWu -WuName $wu)) {
                if (-not $pending.ContainsKey($rname)) {
                    $pending[$rname] = $false
                }
            }
        }
        $stillPending = $pending.GetEnumerator() | Where-Object { -not $_.Value } | ForEach-Object { $_.Key }
        Write-Host ("[t+{0}s] Pending results: {1}" -f $elapsed, (($stillPending) -join ", "))
        foreach ($resultName in @($pending.Keys)) {
            if ($pending[$resultName]) { continue }
            $state = Get-ResultState -ResultName $resultName
            if ($state) {
                Write-Host ("{0} state server={1} outcome={2} validate={3} assimilate={4} file_delete={5}" -f $resultName, $state.ServerState, $state.Outcome, $state.ValidateState, $state.AssimilateState, $state.FileDeleteState)
            }
            if ($state -and $state.ServerState -eq 5 -and $state.ValidateState -eq 1 -and $state.AssimilateState -eq 2) {
                $pending[$resultName] = $true
            }
        }
        if ($pending.Values -notcontains $false) {
            break
        }
    }

    if ($pending.Count -eq 0) {
        Write-Warning "No result records were created for round $r within the polling window."
        Pop-Location
        exit 1
    }

    $incomplete = $pending.GetEnumerator() | Where-Object { -not $_.Value } | ForEach-Object { $_.Key }
    if ($incomplete.Count -gt 0) {
        Write-Warning "Round $r did not finish within $PollSeconds seconds. Pending results: $($incomplete -join ', ')"
        Pop-Location
        exit 1
    }

    $resultNames = @($pending.Keys)
    foreach ($resultName in $resultNames) {
        $listCmd = "ls -1 /results/cifar100/$resultName 2>/dev/null"
        $dirExists = docker compose exec -T apache bash -lc "test -d /results/cifar100/$resultName && echo 1 || echo 0"
        $files = $null
        Write-Host "Result $resultName validated and assimilated by BOINC daemons."
        if ($dirExists -eq "1") {
            $files = docker compose exec apache bash -lc $listCmd
            if ($files) {
                Write-Host "Result artifacts under /results/cifar100/$resultName inside the apache container:"
                $files | ForEach-Object { Write-Host " - $_" }
            } else {
                Write-Host "No files found under /results/cifar100/$resultName; checking upload cache next."
            }
        } else {
            Write-Warning "Result directory /results/cifar100/$resultName missing; checking upload cache."
        }

        $localDir = Join-Path $PSScriptRoot "results_local/cifar100/$resultName"
        New-Item -ItemType Directory -Force -Path $localDir | Out-Null
        if ($dirExists -eq "1") {
            try {
                docker compose cp apache:/results/cifar100/$resultName/. $localDir 2>$null | Out-Null
            } catch {
                Write-Warning "Failed to copy container results for ${resultName}: $_"
            }
        }

        $uploadPath = docker compose exec -T apache bash -lc "find /home/boincadm/project/upload -maxdepth 3 -type f -name '${resultName}_*' | head -n1"
        if ($uploadPath) {
            $metricTarget = Join-Path $localDir "metrics.json"
            try {
                $uploadData = docker compose exec -T apache bash -lc "cat $uploadPath" 2>$null
                Set-Content -Path $metricTarget -Value $uploadData -Encoding UTF8
            } catch {
                Write-Warning "Failed to copy upload file $uploadPath for ${resultName}: $_"
            }
        }

        $metricFiles = Get-ChildItem -Path $localDir -Filter "metrics*.json" -File -ErrorAction SilentlyContinue
        if ($metricFiles) {
            $metric = Get-Content -Raw $metricFiles[0].FullName | ConvertFrom-Json
            $roundSummaries += [pscustomobject]@{
                Round     = $r
                Result    = $resultName
                TrainAcc  = [double]$metric.train_acc
                ValAcc    = [double]$metric.val_acc
                ValTop5   = [double]$metric.val_top5
                TrainLoss = [double]$metric.train_loss
                ValLoss   = [double]$metric.val_loss
                RuntimeM  = [math]::Round($metric.runtime_sec / 60, 2)
            }
            Write-Host ("Metrics for {0}: train_acc={1:F3} val_acc={2:F3} val_top5={3:F3} runtime={4}m" -f $resultName, $metric.train_acc, $metric.val_acc, $metric.val_top5, [math]::Round($metric.runtime_sec/60,2))
        } else {
            Write-Warning "No metrics JSON copied for $resultName"
        }
    }
}

$roundSummaries | Sort-Object -Property Round | Format-Table -AutoSize

Pop-Location
