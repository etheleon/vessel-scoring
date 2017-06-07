#!/bin/bash

python scripts/add_measures.py ../training-data/data/merged/kristina_trawl.npz datasets/kristina_trawl.measures.npz
python scripts/add_measures.py ../training-data/data/merged/kristina_ps.npz datasets/kristina_ps.measures.npz
python scripts/add_measures.py ../training-data/data/merged/kristina_longliner.npz datasets/kristina_longliner.measures.npz

python scripts/add_measures.py ../training-data/data/merged/false_positives.npz datasets/false_positives.measures.npz
python scripts/add_measures.py ../training-data/data/merged/alex_crowd_sourced.npz datasets/classified-filtered.measures.npz
