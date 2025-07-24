#!/usr/bin/env bash

# Loop over all .f90 files in the current directory
for file in *.f90; do
	# Skip if no .f90 files are found
	[ -e "$file" ] || continue

	# Remove the .f90 extension and append .txt
	new_name="${file%.f90}.txt"

	# Rename the file
	cp "$file" "../txt/$new_name"

	echo "Renamed '$file' to '$new_name'"
done
