# Register (or with -Remove, unregister) a Windows logon task that runs start_all.ps1.
# Runs only while the user is logged on, so no password is stored.
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 [-Remove]
param([switch]$Remove)

$TaskName = "SwingCoach Autostart"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    "removed $TaskName"
    return
}

$script = Join-Path $PSScriptRoot "start_all.ps1"
$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$trigger.Delay = "PT30S"  # let the desktop and network settle first
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
# The script only launches processes and waits for ports; they keep running after it exits.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description "Start Swing Coach (Ollama, platform, app, CDC stream) at logon" -Force |
    Select-Object TaskName, State
