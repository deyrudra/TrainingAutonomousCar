param (
  [int]$Duration = 1500,
  [string[]]$Maps = @('Town06','Town05','Town04','Town03','Town02'),
  [string[]]$Parts = @('-1','-2','-3')
)

foreach ($suffix in $Parts) {
  foreach ($town in $Maps) {
    if ($suffix -eq '') {
      $prefix = "${town}_${Duration}s"
    } else {
      $prefix = "${town}_${Duration}s${suffix}"
    }

    Write-Host "=== Waiting before run for $town $suffix ==="
    Start-Sleep -Seconds 5  # <-- Add desired delay here

    Write-Host "=== Running on $town (output: $prefix) ==="
    python DataCollection.py --duration $Duration --map $town --prefix $prefix
  }
}

Write-Host "All runs complete."
