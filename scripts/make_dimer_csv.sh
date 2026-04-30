#!/bin/bash

DIR="/home/jcr222/scratch_pi_sk2433/hm638/dimer_filtered"
OUTPUT="${1:-dimer_filtered.csv}"

echo ",identifier,cif_path" > "$OUTPUT"

i=0
for f in "$DIR"/*.cif; do
    base=$(basename "$f")
    identifier=$(echo "$base" | sed 's/^AF-//; s/-model_v[0-9]*\.cif$//')
    echo "$i,$identifier,$f" >> "$OUTPUT"
    ((i++))
done

echo "Done. $i entries written to $OUTPUT"
