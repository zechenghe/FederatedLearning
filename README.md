An implementation of Federated Learning, a.k.a Collaborative Learning.

#### Dependencies:

python 2.7
```
python --version
```

numpy 1.16.5

```
import numpy as np
print np.__version__
```

pytorch 1.3.0

```
import torch
print torch.__version__
```

torchvision 0.4.1

```
import torchvision
print torchvision.__version__
```

scikit-image 0.14.5

```
import skimage
print skimage.__version__
```

#### Run federated learning

python collaborative.py --n_clients 4
Use --nogpu if you do not have GPU on hand
