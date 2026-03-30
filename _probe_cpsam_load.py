print('a', flush=True)
from cellpose import io, models
print('b', flush=True)
img = io.imread(r'.\\images\\cellpose.png')
print('img', img.shape, img.dtype, flush=True)
m = models.CellposeModel(gpu=False, pretrained_model='cpsam')
print('model loaded', flush=True)
