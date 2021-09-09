# REPOS

git clone git@github.com:gtatiya/sound-spaces.git

git clone git@github.com:jonfranc/neural-avn.git

git clone git@github.com:facebookresearch/habitat-sim.git

git clone git@github.com:facebookresearch/habitat-lab.git

# CONDA 

conda create -n avn python=3.6 cmake=3.14.0
conda activate avn

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

conda install habitat-sim=0.1.7 headless -c conda-forge -c aihabitat
conda install h5py

cd habitat-lab
git fetch --all --tags
git checkout tags/v0.1.7
pip install -e .
python setup.py develop --all


