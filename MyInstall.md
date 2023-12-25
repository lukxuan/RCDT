conda create -n pie_cd python=3.8 -y
conda activate pie_cd
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/ 
# 设置搜索时显示通道地址conda config --set show_channel_urls yes

# pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f
# https://download.pytorch.org/whl/cu102/torch_stable.html

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

conda install gdal
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install geopandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv>=2.0.0rc1" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "mmdet>=2.0.0rc6" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "mmsegmentation>=1.0.0rc0" -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# '-v' 表示详细模式，更多的输出
# '-e' 表示以可编辑模式安装工程，
# 因此对代码所做的任何修改都生效，无需重新安装

pip install ipdb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install future tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install rasterio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install shapely -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seaborn progress

# 问题
1，mmcv 2.0.0rc4 移除了 self.file_client，导致“assert self.file_client.exists(self.data_prefix['img_path'])”这行代码会导致bug;
2,出现registry错误一般是注册器混淆piecd、mmseg、mmdet导致的
3,instcdnet和instmaskcdnet对于num_quries参数设置敏感度不一样，后者只适合2