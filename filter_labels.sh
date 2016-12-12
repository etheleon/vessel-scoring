ls datasets/data/labeled |
  grep measures.npz |
  while read name; do
    echo "Filtering $name..."
    scripts/filter_labels.py datasets/data/labeled/$name datasets/data/labeled/$(basename $name .npz).labels.npz
  done
