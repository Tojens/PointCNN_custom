#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import plyfile
import numpy as np
import argparse
import h5py

reduced_length_dict = {"MarketplaceFeldkirch":[10538633,"marketsquarefeldkirch4-reduced"],
                       "StGallenCathedral":[14608690,"stgallencathedral6-reduced"],
                       "sg27":[28931322,"sg27_10-reduced"],
                       "sg28":[24620684,"sg28_2-reduced"]}

full_length_dict = {"6725_66515_no_color":[475136, "6725_66515_no_color"],
                    "6725_66520_no_color":[24576, "6725_66520_no_color"],
                    "6730_66515_no_color":[540672, "6730_66515_no_color"],
                    "6730_66520_no_color":[286720, "6730_66520_no_color"],
                    "6735_66515_no_color":[344064, "6735_66515_no_color"],
                    "6735_66520_no_color":[950272, "6735_66520_no_color"],
                    "6740_66520_no_color":[942080, "6740_66520_no_color"],
                    "6745_66520_no_color":[917504, "6745_66520_no_color"],
                    "6745_66525_no_color":[991232, "6745_66525_no_color"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    parser.add_argument('--version', '-v', help='full or reduced', type=str, required=True)
    args = parser.parse_args()
    print(args)

    if args.version == 'full':
        length_dict = full_length_dict
    else:
        length_dict = reduced_length_dict

    categories_list = [category for category in length_dict]
    #print(categories_list)

    for category in categories_list:
        output_path = os.path.join(args.datafolder,"results",length_dict[category][1]+".labels")
        if not os.path.exists(os.path.join(args.datafolder,"results")):
            os.makedirs(os.path.join(args.datafolder,"results"))
        pred_list = [pred for pred in os.listdir(args.datafolder)
                     if category in pred  and pred.split(".")[0].split("_")[-1] == 'pred']

        label_length = length_dict[category][0]
        merged_label = np.zeros((label_length),dtype=int)
        merged_confidence = np.zeros((label_length),dtype=float)

        for pred_file in pred_list:
            #print(os.path.join(args.datafolder, pred_file))
            data = h5py.File(os.path.join(args.datafolder, pred_file))
            labels_seg = data['label_seg'][...].astype(np.int64)
            indices = data['indices_split_to_full'][...].astype(np.int64)
            confidence = data['confidence'][...].astype(np.float32)
            data_num = data['data_num'][...].astype(np.int64)
            print("file:", data)
            print("size", indices.size)
            for i in range(labels_seg.shape[0]):
                temp_label = np.zeros((data_num[i]),dtype=int)
                pred_confidence = confidence[i][:data_num[i]]
                temp_confidence = merged_confidence[indices[i][:data_num[i]]]
                
                temp_label[temp_confidence >= pred_confidence] = merged_label[indices[i][:data_num[i]]][temp_confidence >= pred_confidence]
                temp_label[pred_confidence > temp_confidence] = labels_seg[i][:data_num[i]][pred_confidence > temp_confidence]

                merged_confidence[indices[i][:data_num[i]][pred_confidence > temp_confidence]] = pred_confidence[pred_confidence > temp_confidence]
                merged_label[indices[i][:data_num[i]]] = temp_label
                #print("indices", indices[0][1].size)
        np.savetxt(output_path,np.c_[indices.flatten(), merged_label+1,merged_confidence],fmt='%d %d %.3f')

if __name__ == '__main__':
    main()
