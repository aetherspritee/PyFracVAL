#!/usr/bin/env bash

rg=1000.0
verbose=0
gstds=(
	1.2
	1.5
)

kf=1.50

ns=(
	32
	# 64
	# 96
	# 128
	# 160
	# 192
	# 224
	# 256
)
dfs=(
	1.4
	1.6
	1.8
	2.0
	2.2
)

if [[ "$verbose" = 1 ]]; then
	v="-v"
elif [[ "$verbose" = 2 ]]; then
	v="-vv"
elif [[ "$verbose" = 3 ]]; then
	v="-vvv"
else
	v=""
fi

for gstd in "${gstds[@]}"; do
	for n in "${ns[@]}"; do
		for df in "${dfs[@]}"; do
			# echo "Generating $n $df $gstd"
			echo "uv run pyfracval -n \"$n\" --df \"$df\" --kf \"$kf\" --rp-g \"$rg\" --rp-gstd \"$gstd\""
			uv run pyfracval -n "$n" --df "$df" --kf "$kf" --rp-g "$rg" --rp-gstd "$gstd" $v
		done
	done
done
