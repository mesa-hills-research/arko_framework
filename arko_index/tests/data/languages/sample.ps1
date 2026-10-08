<#
.SYNOPSIS
  Deploys the service.
#>
param(
    [string]$Environment = "staging",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# Writes a log line.
function Write-Log {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format o)] $Message"
}

function Get-Config([string]$Name) {
    $path = Join-Path $PSScriptRoot "config/$Name.json"
    return Get-Content $path | ConvertFrom-Json
}

class Deployment {
    [string]$Target
    [int]$Retries = 3

    Deployment([string]$target) {
        $this.Target = $target
    }

    [bool] Run() {
        for ($i = 0; $i -lt $this.Retries; $i++) {
            Write-Log "deploying to $($this.Target), attempt $i"
            if (Test-Connection $this.Target -Quiet) { return $true }
        }
        return $false
    }
}

$config = Get-Config $Environment
$deployment = [Deployment]::new($config.Host)

if (-not $WhatIf) {
    if (-not $deployment.Run()) {
        throw "deployment failed"
    }
}

foreach ($service in $config.Services) {
    Write-Log "restarting $service"
}
