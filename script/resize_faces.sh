#!/usr/bin/env bash
mogrify -resize $1x$1! -quality 100 raw_data/*.jpg
mogrify -resize $1x$1! -quality 100 raw_data/*.png
