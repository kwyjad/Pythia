Param(
  [string]$WheelDir = "tools/offline_wheels"
)

python -m pip install --upgrade pip
python -m pip install --no-index --find-links $WheelDir -r "$WheelDir/constraints-db.txt"
