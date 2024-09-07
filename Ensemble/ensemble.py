import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--J_Score', 
        type = str,
        default = './Score/NTU120_XSub_J.pkl') # J 
    parser.add_argument(
        '--B_Score', 
        type = str,
        default = './Score/NTU120_XSub_B.pkl') # B
    parser.add_argument(
        '--JM_Score', 
        type = str,
        default = './Score/NTU120_XSub_JM.pkl') # JM
    parser.add_argument(
        '--BM_Score', 
        type = str,
        default = './Score/NTU120_XSub_BM.pkl') # BM
    parser.add_argument(
        '--HDJ_Score', 
        type = str,
        default = './Score/NTU120_XSub_HDJ.pkl') # HDJ
    parser.add_argument(
        '--HDB_Score', 
        type = str,
        default = './Score/NTU120_XSub_HDB.pkl') # HDB
    parser.add_argument(
        '--val_sample', 
        type = str,
        default = './Val_Sample/NTU120_XSub_Val.txt') # NTU120_XSub_Val.txt
    parser.add_argument(
        '--benchmark', 
        type = str,
        default = 'NTU120XSub')
    return parser

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx in range(6):
        fr = open(File[idx],'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label_ntu(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype = str)
    for idx, name in enumerate(val_txt):
        label = int(name[-3:]) - 1
        true_label.append(label)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    j_file = args.J_Score
    b_file = args.B_Score
    jm_file = args.JM_Score
    bm_file = args.BM_Score
    hdj_file = args.HDJ_Score
    hdb_file = args.HDB_Score
    val_txt_file = args.val_sample

    File = [j_file, b_file, jm_file, bm_file, hdj_file, hdb_file] 
    if args.benchmark == 'NTU60XSub':
        Rate = [0., 0., 0., 0., 0.]
        Numclass = 60
        Sample_Num = 16487
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)
    
    elif args.benchmark == 'NTU60XView':
        Rate = [0., 0., 0., 0., 0.]
        Numclass = 60
        Sample_Num = 18932
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)
    
    elif args.benchmark == 'NTU120XSub':
        Rate = [2.361693262051476, 4.35402786600752, 3.6038236890242237, 1.0158095311333564, 3.0574845840597544, 5.0]
        Numclass = 120
        Sample_Num = 50919
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)
    
    elif args.benchmark == 'NTU120XSet':
        Rate = [0., 0., 0., 0., 0.]
        Numclass = 120
        Sample_Num = 59477
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label_ntu(val_txt_file)

    Acc = Cal_Acc(final_score, true_label)

    print('acc:', Acc)
    
    print("All Done!")