conda create -n opara python=3.10 -y
conda activate opara
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e .