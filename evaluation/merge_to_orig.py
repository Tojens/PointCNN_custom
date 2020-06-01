# Import modules
import h5py
import numpy as np

# Load in h5 file (just as an example)
pred_file = '//mnt/edisk/PointCNN/data/semantic3d/train/6750_66520_no_color_zero_0.h5'

# Load in h5 files
data = h5py.File(pred_file)

#img = data['data'][...]
data_num = data['data_num'][...]
indices = data['indices_split_to_full'][...]
label_seg = data['label_seg'][...]
#confidence = data['confidence'][...]
print(indices.size)
max_ind = np.max(indices) # Get max index
label_flat = -1 * np.ones(max_ind, dtype=np.int32) # I make it -1 since a label of '0' is an actual label

label_flat[indices.flatten()-1]= label_seg.flatten() # indicies.flatten() - 1 because the index is out of range with indices.flatten()
print(label_flat.size)