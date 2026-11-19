#!/usr/bin/env bash
# Download the DeepLense GSoC test datasets (links from the public 2026 test document).
# Common Test I (1.0 GB) + Specific Test VI.A (509 MB) + VI.B (21 MB).
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
get () {  # get <drive-file-id> <dir> <name>
  echo "== $3 -> $2"
  curl -L --fail --retry 3 -C - -o "$2/$3" \
    "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"
  file "$2/$3"
}
get 1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ common     dataset.zip
get 1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw superres_a Dataset.zip
get 1plYfM-jFJT7TbTMVssuCCFvLzGdxMQ4h superres_b Dataset3B.zip
echo "Done. Unzip each one inside its own folder, e.g.  (cd common && unzip -q dataset.zip)"
df -h / | tail -1
