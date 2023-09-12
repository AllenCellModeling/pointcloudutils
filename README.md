# pointcloudutils

Utils and networks for point cloud related projects at AICS

---

First install cyto_dl. Then 

```bash
pip install -e .

pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

To use
 
```bash

from im2mesh.utils.libmise.mise import MISE
from im2mesh.utils.libmcubes import marching_cubes
```

install the occupancy networks repo and build these 2 extensions

**Allen Institute Software License**

