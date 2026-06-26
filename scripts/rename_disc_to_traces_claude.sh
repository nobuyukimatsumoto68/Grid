#!/bin/bash

dir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b11p045_m0p4000"

for f in "${dir}"/disc.*; do
  base=$(basename "$f")
  newname="${base/disc./traces.}"
  mv "$f" "${dir}/${newname}"
done

echo "Done"
