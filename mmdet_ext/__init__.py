from .datasets.pipelines import *
from .datasets import *
from .models.detectors import *

# since mmdetection use decorator to register cls to cls factory, need to import cls to process decorator
print("import mmdet_ext successfully")