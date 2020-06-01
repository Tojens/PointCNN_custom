import numpy as np
import os
import h5py
import pandas as pd
data_folder = "/mnt/edisk/PointCNN/data/semantic3d/val/iou_test"

pred_list = [pred for pred in os.listdir(data_folder) \
         if pred.split(".")[0].split("_")[-1] == 'pred']

#Data folder is where my predicted and tested h5 files are located
acc, tot = 0,0
result = np.zeros((6,6),dtype=int) #result is the confusion matrix
max_ind = 0 #Calculate max indices


for pred in pred_list:

    data = h5py.File(os.path.join(data_folder, pred))
   
    f = '_'.join(pred.split('_')[:-1])+'.h5' #Open corresponding test h5 file 
    data_test = h5py.File(os.path.join(data_folder, f))

    # Open predicted h5 file

    labels_seg = data['label_seg'][...].astype(np.int64)
    indices = data['indices_split_to_full'][...].astype(np.int64)
    if indices.max() > max_ind: max_ind = indices.max() 
    confidence = data['confidence'][...].astype(np.float32)
    data_num = data['data_num'][...].astype(np.int64)
    print(indices.size)
    # Open test h5 file

    t_labels_seg = data_test['label_seg'][...].astype(np.int64)
    t_indices = data_test['indices_split_to_full'][...].astype(np.int64)
    t_data_num = data_test['data_num'][...].astype(np.int64)
    print(t_indices.size)
 # Loop through corresponding h5 file and calculate confusion matrix

    for i in range(labels_seg.shape[0]):
        test = t_labels_seg[i][:t_data_num[i]]
        predicted = labels_seg[i][:data_num[i]]
        test_ind = t_indices[i][:t_data_num[i]]
        ind = indices[i][:data_num[i]]
        if False in np.equal(test_ind,ind): print('Indices don\'t match!') # Just a sanity check to ensure the indices match
        
        tot += test.shape[0]
        dif = test==predicted
        
        acc+=dif.sum()

### Calculate confusion matrix
        for i in range(len(predicted)):
            result[test[i]][predicted[i]] += 1
        
        #print(result)

        #print(dif)
        #iou_list = []
        #gt_classes = test
        #positive_classes = predicted
        #true_positive_classes = dif==True
        #print("gt_classes", gt_classes[i])
        #print("positive_classes", positive_classes[i])
        #print("true_positive_classes", true_positive_classes[i])

        
        #print(true_positive_classes)

        #for i in range(len(predicted)):
         #   iou = true_positive_classes[i]/(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
         #   #print("  {}: {}".format(i, iou))
         #   samlet_iou=iou_list.append(iou)
         #   samlet_iou1=iou_list
         #   
         #   samlet_iou2=[]
         #   samlet_iou2.append(samlet_iou1)
         #   print("Average IoU: {}".format(sum(iou_list)/len(predicted)))
            #print(i)
            #print(samlet_iou1)
            #print(len(samlet_iou1))
            #print(len(samlet_iou2))
         #   print(data_test)
        #print("new")

    data.close()
    data_test.close()