import numpy as np
import scipy
from adjacencyMat import adjacencyMat

def midPointUpsampling(V,F,numIter=1):
    '''
    midPointUpsampling do mid point upsampling 

    Inputs:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
        numIter number of upsampling to perform

    Outputs:
        V |V|-by-3 numpy array of new vertex positions
        F |F|-by-3 numpy array of new face indices
        upOpt |Vup|-by-|V| numpy array of upsampling operator

    TODO:
        add boundary constraints 
    '''
    for iter in range(numIter):
        nV = V.shape[0]
        nF = F.shape[0]

        # compute new vertex positions
        hE = np.concatenate( (F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]), axis=0 )  #把面的三条边都提取出来拼成一个数组。维度应该是[face numbers*3,2]
        hE = np.sort(hE, axis = 1) # 对于每一行，即每一条边，把顶点索引号小的顶点放前面
        E, hE2E = np.unique(hE, axis=0, return_inverse=True)  # 去除重复的边，返回的E是将边的第0个顶点索引从小到大排列的边list, return_inverse为true，返回旧列表hE元素在新列表E中的位置（下标），并以列表形式存储。
        nE = E.shape[0] #真正的边的数量
        newV = (V[E[:,0],:] + V[E[:,1],:]) / 2.0 #插入中点的位置，中点的数量和边的数量一样
        V = np.concatenate( (V, newV), axis = 0 ) #新的顶点加入到老的顶点后面，构成新的顶点list V

        # compute updated connectivity
        i2 = nV       + np.arange(nF) # 初始化一个array[nV，nV+1,...,nV+nF]
        i0 = nV+nF    + np.arange(nF)
        i1 = nV+nF+nF + np.arange(nF)

        hEF0 = np.concatenate( (F[:,0:1], i2[:,None], i1[:,None]), axis=1 ) # 数组维度[nF,3],每一行的第一位来自F,第二位来自i2
        hEF1 = np.concatenate( (F[:,1:2], i0[:,None], i2[:,None]), axis=1 )
        hEF2 = np.concatenate( (F[:,2:3], i1[:,None], i0[:,None]), axis=1 )
        hEF3 = np.concatenate( (i0[:,None], i1[:,None], i2[:,None]), axis=1 )
        hEF  = np.concatenate( (hEF0, hEF1, hEF2, hEF3), axis=0 ) #数组维度[nF*4,3] ，1个面变4个面

        hE2E =  np.concatenate( (np.arange(nV), hE2E+nV), axis=0 ) # 顶点数量+旧边的数量[0,1,..,nV-1,hE2E+nV]，旧边的数量是面数*3
        uniqV = np.unique(F) # 去除F中的重复元素，返回的是旧列表元素在新列表中的位置（下标）
        F = hE2E[hEF] #F和hEF一样的维度，以hEF中每一个行是一个面的三个顶点索引[v0,v1,v2]，以这个顶点索引去取hE2E中该索引下的值，即[hE2E[v0],hE2E[v1],hE2E[v2]]。

        # upsampling for odd vertices
        rIdx = uniqV
        cIdx = uniqV
        val = np.ones((len(uniqV),))

        # upsampling for even vertices
        rIdx = np.concatenate( (rIdx, nV+np.arange(nE),  nV+np.arange(nE)) )
        cIdx = np.concatenate( (cIdx, E[:,0],  E[:,1]) )
        val = np.concatenate( (val, np.ones(2*nE)*0.5) )

        # upsampling operator
        if iter == 0:
            S = scipy.sparse.coo_matrix( (val, (rIdx,cIdx)), shape = (nV+nE, nV) ) # (val, (rIdx,cIdx)=(data,(row,col))，上采样后的所有顶点是由哪两个老顶点计算得来的。
        else:
            tmp = scipy.sparse.coo_matrix( (val, (rIdx,cIdx)), shape = (nV+nE, nV) )
            S = tmp * S

    return V, F, S