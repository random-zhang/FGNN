import math

import numpy as np
import pandas as pd


def NMI(A,B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)    # 输出满足条件的元素的下标
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)   # Find the intersection of two arrays.
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
        Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    if Hx+Hy==0:
        MIhat= np.nan
    else:
        MIhat = 2.0*MI/(Hx+Hy)
    return MIhat


def CMI(x, y, z):
    # 初始化条件互信息mi
    mi = 0
    # 获取不重复的 (x,y,z)
    xyz_unique = np.unique(np.column_stack((x, y, z)), axis=0)
    # 遍历每一个不同的 (x,y,z)
    for xyz in xyz_unique:
        # 统计 (x,y,z)出现的次数count_xyz
        count_xyz = np.sum((x == xyz[0]) & (y == xyz[1]) & (z == xyz[2]))
        # 计算 (x,y,z)出现的概率p_xyz
        p_xyz = count_xyz / len(x)
        # 统计z出现的次数count_z
        count_z = np.sum(z == xyz[2])
        # 计算条件概率：在当前的z下 (x,y)出现的概率p_xy|z
        p_xy_z = count_xyz / count_z
        # 计算条件概率：在当前的z下x出现的概率p_x|z
        count_xz = np.sum((x == xyz[0]) & (z == xyz[2]))
        p_x_z = count_xz / count_z
        # 计算条件概率：在当前的z下y出现的概率p_y|z
        count_yz = np.sum((y == xyz[1]) & (z == xyz[2]))
        p_y_z = count_yz / count_z
        mi += p_xyz * np.log(p_xy_z / (p_x_z * p_y_z))
    return mi
