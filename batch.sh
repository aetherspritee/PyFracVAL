#!/usr/bin/env bash

rg=1000.0
gstds=(1.2 1.5)

kf=1.50

ns=(
	32
	64
	96
	128
	160
	192
	224
	256
)
dfs=(1.4 1.6 1.8 2.0 2.2)

for gstd in "${gstds[@]}"; do
	for n in "${ns[@]}"; do
		for df in "${dfs[@]}"; do
			# echo "Generating $n $df $gstd"
			echo "uv run pyfracval -n \"$n\" --df \"$df\" --kf \"$kf\" --rp-g \"$rg\" --rp-gstd \"$gstd\""
			uv run pyfracval -n "$n" --df "$df" --kf "$kf" --rp-g "$rg" --rp-gstd "$gstd"
		done
	done
done

# pyfracval -n 8 --df 1.4 --kf 1.25 --rp-std 5
# pyfracval -n 8 --df 1.6 --kf 1.25 --rp-std 5
# pyfracval -n 8 --df 1.8 --kf 1.25 --rp-std 5
# pyfracval -n 8 --df 2.0 --kf 1.25 --rp-std 5
# pyfracval -n 8 --df 2.2 --kf 1.25 --rp-std 5

# pyfracval -n 20 --df 1.4 --kf 1.25 --rp-std 5
# pyfracval -n 20 --df 1.6 --kf 1.25 --rp-std 5
# pyfracval -n 20 --df 1.8 --kf 1.25 --rp-std 5
# pyfracval -n 20 --df 2.0 --kf 1.25 --rp-std 5
# pyfracval -n 20 --df 2.2 --kf 1.25 --rp-std 5

# pyfracval -n 48 --df 1.4 --kf 1.25 --rp-std 5
# pyfracval -n 48 --df 1.6 --kf 1.25 --rp-std 5
# pyfracval -n 48 --df 1.8 --kf 1.25 --rp-std 5
# pyfracval -n 48 --df 2.0 --kf 1.25 --rp-std 5
# pyfracval -n 48 --df 2.2 --kf 1.25 --rp-std 5

# pyfracval -n 80 --df 1.4 --kf 1.25 --rp-std 5
# pyfracval -n 80 --df 1.6 --kf 1.25 --rp-std 5
# pyfracval -n 80 --df 1.8 --kf 1.25 --rp-std 5
# pyfracval -n 80 --df 2.0 --kf 1.25 --rp-std 5
# pyfracval -n 80 --df 2.2 --kf 1.25 --rp-std 5
