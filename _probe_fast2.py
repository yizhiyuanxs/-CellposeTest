import time
from cellpose import io, models
img=io.imread(r'.\\images\\cellpose.png')
m=models.CellposeModel(gpu=False, pretrained_model='cpsam')
t=time.time()
out = m.eval(img, batch_size=1, compute_masks=False)
print('done', round(time.time()-t,2))
