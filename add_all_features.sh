ls datasets/data/labeled |
  grep -v measures |
  while read name; do
    dstname="datasets/data/labeled/$(basename $name .npz).measures.npz"
    name="datasets/data/labeled/$name"
    if [ -e $dstname ]; then
      echo "$dstname exists"
    else
      echo "Add features $name => $dstname"
      python scripts/add_features.py $name $dstname
    fi
  done
