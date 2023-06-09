mkdir tests\data

curl https://raw.githubusercontent.com/themis-ai/capsa/main/test/data/mve_max_points.npy --output tests\data\mve_max_points.npy
curl https://raw.githubusercontent.com/themis-ai/capsa/main/test/data/mve_min_points.npy --output tests\data\mve_min_points.npy

curl https://raw.githubusercontent.com/themis-ai/capsa/main/test/data/ensemble_max_points.npy --output tests\data\ensemble_max_points.npy
curl https://raw.githubusercontent.com/themis-ai/capsa/main/test/data/ensemble_min_points.npy --output tests\data\ensemble_min_points.npy
