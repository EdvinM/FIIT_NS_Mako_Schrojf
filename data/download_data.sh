#!/usr/bin/env bash

wget -O imdb_meta.tar https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar
mkdir -p raw/imdb_meta
tar -xvf imdb_meta.tar -C raw/imdb_meta/
rm imdb_meta.tar

#TODO:
#https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
#https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz
#https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar