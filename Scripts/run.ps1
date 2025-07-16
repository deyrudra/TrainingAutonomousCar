param (
  [int]$Duration = 80,
  [string[]]$Maps = @('Town07','Town06','Town05','Town04','Town03','Town02','Town01'),
  [string[]]$Parts = @('','-2','-3','-4')
)

foreach ($suffix in $Parts) {
  foreach ($town in $Maps) {
    if ($suffix -eq '') {
      $prefix = "${town}_${Duration}s"
    } else {
      $prefix = "${town}_${Duration}s${suffix}"
    }
    Write-Host "=== Running on $town (output: $prefix) ==="
    python DataCollection.py --duration $Duration --map $town --prefix $prefix
  }
}


Write-Host "All runs complete."