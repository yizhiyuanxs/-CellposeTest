import time
print('a', flush=True)
from cellpose import io, models

print('b', flush=True)
img = io.imread(r'.\\images\\cellpose.png')
print('img', img.shape, img.dtype, flush=True)
m = models.CellposeModel(gpu=False, pretrained_model='cpsam')
print('c', flush=True)
t = time.time()
masks, flows, styles = m.eval(img, diameter=None, batch_size=8, flow_threshold=0.4, cellprob_threshold=0.0)
print('d', round(time.time()-t,2), flush=True)
print('masks', masks.shape, masks.dtype, flush=True)
