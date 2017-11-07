import h5py, os
import numpy as np

SIZE = 227 # fixed size to all images

X = np.ones((1, 3, SIZE, SIZE), dtype='f8')

with h5py.File('test_idty.h5','w') as H:
    H.create_dataset('img', data=X )
with open('test_h5_idty_list.txt','w') as L:
    L.write( '/home/wei/deep_metric/test_idty.h5' )


filename = 'test_idty.h5'
f = h5py.File(filename, 'r')

print list(f['img'])