# Restore only the banner values changed for the 2026-10-24 SLM experiment.
# This does not change notification delivery, lock, or security settings.
$paperRepository = Split-Path -Parent $PSScriptRoot
$paperRun = Join-Path $paperRepository 'results\paper_cnnpropcnn_20261024'
$paperBackups = @('notification_banner_backup.json','notification_smartoptout_backup.json')
foreach ($paperBackupName in $paperBackups) {
    $paperBackupPath = Join-Path $paperRun $paperBackupName
    $paperStates = @(Get-Content -LiteralPath $paperBackupPath -Raw | ConvertFrom-Json)
    foreach ($paperState in $paperStates) {
        $paperRegistryPath = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Notifications\Settings\' + $paperState.name
        $paperCurrent = Get-ItemProperty -LiteralPath $paperRegistryPath
        if ($paperCurrent.ShowBanner -ne 0) {
            Write-Warning "Banner value changed since experiment; preserving current setting for $($paperState.name)"
            continue
        }
        if ($paperState.existed) {
            New-ItemProperty -LiteralPath $paperRegistryPath -Name ShowBanner -Value $paperState.value -PropertyType DWord -Force | Out-Null
        } else {
            Remove-ItemProperty -LiteralPath $paperRegistryPath -Name ShowBanner
        }
    }
}
