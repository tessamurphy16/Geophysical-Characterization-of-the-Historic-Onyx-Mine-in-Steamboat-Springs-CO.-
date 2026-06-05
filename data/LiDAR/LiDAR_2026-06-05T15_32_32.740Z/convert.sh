#!/bin/zsh
for file in *.img
do
  fname="$(basename $file .img)"
  gdal_translate -of GTiff ${fname}.img ${fname}.tif
done
