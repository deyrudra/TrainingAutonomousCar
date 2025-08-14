#powershell script to automoate the datacollection across various maps
#here we contorl the duration, along with which maps data is collected on
param (
  [int]$Duration = 400, # time change between large and small datasets
  [string[]]$Maps = @('Town10HD','Town07','Town06','Town05','Town04','Town03','Town02','Town01'),
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
    Start-Sleep -Seconds 5

    Write-Host "=== Running on $town (output: $prefix) ==="
    python DataCollection.py --duration $Duration --map $town --prefix $prefix
  }
}

Write-Host "All runs complete."
