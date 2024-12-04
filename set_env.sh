conda create -n stream python=3.10 -y
conda activate stream
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .