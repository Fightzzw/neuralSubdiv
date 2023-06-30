import os
import sys

# cwd = os.getcwd()
# sys.path.append('../pythongptoolbox/')
# sys.path.append('../torchgptoolbox_nosparse/')
# sys.path.append('/work/Users/zhuzhiwei/neuralSubdiv-master/pythongptoolbox/')
# sys.path.append('/work/Users/zhuzhiwei/neuralSubdiv-master/torchgptoolbox_nosparse/')

# torch
import torch
import torchgptoolbox_nosparse as tgp

# pygptoolbox
from pythongptoolbox.midPointUpsampling import midPointUpsampling

# python standard 
import numpy as np
import scipy
import scipy.sparse
import glob
import time
import sys
import json
import pickle as pickle


class Mesh:
    # store information of a mesh
    def __init__(self, V, F, hfList):
        """
        Inputs:
            V: nV-3 vertex list
            F: nF-3 face list
            hfList: nHF-4 ordered vertex index of all half flaps 
            
            Notes: 
            each half flap order looks like (see paper for the color scheme)
            [v_blue, v_red, v_purple, v_yellow]
        """
        self.V = V
        self.F = F
        self.hfList = hfList


def processTrainShapes(folder):
    """
    process training shapes given a folder, including computing the half flap list and read the vertex/face lists
    """
    # detect number of subd and number of meshes per subd
    sub_Folders = os.listdir(folder)
    subFolders = []
    for x in sub_Folders:
        if type(x) == bytes:
            # print('x is bytes')
            x = x.decode('utf-8')
        if not x.startswith('.'):
            subFolders.append(x)
    # print(subFolders)
    nSubd = len(subFolders) - 1
    # nSubd = 2

    objPaths = glob.glob(os.path.join(folder, subFolders[0]) + "/*.obj")
    nObjs = len(objPaths)
    # nObjs = 30

    # create data loading list of strings (glod results are not sorted)
    paths = []
    for ii in range(nSubd + 1):
        paths.append(os.path.join(folder, 'subd' + str(ii) + '/'))

    objFiles = []
    for ii in range(nObjs):
        objFiles.append(str(ii + 1).zfill(len(str(nObjs))))
        # objFiles.append(str(ii + 1).zfill(nSubd))

    # load data
    meshes = []

    for ii in range(nObjs):
        print('\tprocess meshes %d / %d' % (ii, nObjs))
        meshes_i = [None] * (nSubd + 1)
        for jj in range(nSubd + 1):
            print('\tprocess subd %d / %d' % (jj, nSubd))
            V, F = tgp.readOBJ(paths[jj] + objFiles[ii] + '.obj')
            if V.shape[0] == 0:
                print('\tV.shape[0] == 0, pass this mesh')
                meshes_i = None
                break
            try:
                V = tgp.normalizeUnitCube(V) * 2
            except IndexError:
                print('\tIndexError, pass this mesh')
                meshes_i = None
                break
            _, hfList = computeFlapList(V, F, 1)
            if hfList is None:
                print('\thfList is None, pass this mesh')
                meshes_i = None
                break
            meshes_i[jj] = Mesh(V, F, hfList[0])
        if meshes_i is None:
            continue
        meshes.append(list(meshes_i))
    print('num Subd: %d, num meshes: %d' % (nSubd, len(meshes)))
    return meshes

class TrainMeshes:
    """
    store information of many training meshes (see gendataPKL.py for usage)
    """

    def __init__(self, folders):
        """
        Inputs:
            folders: list of folders that contain the meshes
        """

        nShape = len(folders)
        self.meshes = []
        for fIdx in range(nShape):
            print('process training shapes, %d/%d ' % (fIdx, nShape))
            print(folders[fIdx])
            meshes = processTrainShapes(folders[fIdx])
            for ii in range(len(meshes)):
                self.meshes.append(meshes[ii])
        self.nM = len(self.meshes)  # number of meshes
        self.nS = len(self.meshes[0])  # number of subdivision levels

        # initialize parameters required during training
        self.hfList = None  # half flap index information
        self.poolMats = None  # vertex one-ring pooling matrices
        self.dofs = None  # vertex degrees of freedom
        self.LCs = None  # vector of differential coordinates

    def removeMesh(self, mIdxList):
        """
        remove a mesh from the list
        Inputs:
            mIdx: mesh index
        """
        for mIdx in mIdxList:
            self.meshes[mIdx] = None
            self.hfList[mIdx] = None
            self.poolMats[mIdx] = None
            self.dofs[mIdx] = None
            self.LCs[mIdx] = None
        self.meshes = [x for x in self.meshes if x is not None]
        self.hfList = [x for x in self.hfList if x is not None]
        self.poolMats = [x for x in self.poolMats if x is not None]
        self.dofs = [x for x in self.dofs if x is not None]
        self.LCs = [x for x in self.LCs if x is not None]

        self.nM = len(self.meshes)

    def appendMeshList(self, meshes):
        """
        append a mesh to the list
        Inputs:
            meshList: list of meshes
        """
        self.meshes.extend(meshes)

        self.nM = len(self.meshes)
        self.nS = len(self.meshes[0])

    def getInputData(self, mIdx):
        """
        get input data for the network
        Inputs: 
            mIdx: mesh index
        """
        input = torch.cat((
            self.meshes[mIdx][0].V,  # vertex positions
            self.LCs[mIdx]),  # vector of differential coordinates
            dim=1)
        return input

    def old_getHalfFlap(self):
        """
        create a list of half flap information, such that (see paper for the color scheme)
        HF[meshIdx][subdIdx] = [v_blue, v_red, v_purple, v_yellow]
        """
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getHalfFlap(self):
        """
        create a list of half flap information, such that (see paper for the color scheme)
        HF[meshIdx][subdIdx] = [v_blue, v_red, v_purple, v_yellow]
        """
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS)
            for jj in range(self.nS):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getFlapPool(self, HF):
        """
        get the matrix for vertex one-ring average pooling (left two sub-figures in Fig.17)
        Inputs:
            HF: half flap list (see self.getHalfFlap())
        """
        nM = len(HF)  # number of meshes
        nS = len(HF[0])  # number of subdivision levels

        poolFlap = [None] * nM
        dof = [None] * nM
        for ii in range(nM):
            poolFlap_ij = [None] * nS
            dof_ij = [None] * nS
            for jj in range(nS):
                hfIdx = HF[ii][jj]
                nV = hfIdx[:, 0].max() + 1

                rIdx = hfIdx[:, 0]
                cIdx = torch.arange(hfIdx.size(0))
                I = torch.cat([rIdx, cIdx], 0).reshape(2, -1)
                val = torch.ones(hfIdx.size(0))
                poolMat = torch.sparse.FloatTensor(I, val, torch.Size([nV, hfIdx.size(0)]))

                rowSum = torch.sparse.sum(poolMat, dim=1).to_dense()

                poolFlap_ij[jj] = poolMat
                dof_ij[jj] = rowSum
            poolFlap[ii] = list(poolFlap_ij)  # one-ring pooling matrix
            dof[ii] = list(dof_ij)  # degrees of freedom per vertex
        return poolFlap, dof

    def getLaplaceCoordinate(self, hfList, poolMats, dofs):
        """
        get the vectors of the differential coordinates (see Fig.18)
        Inputs:
            hfList: half flap list (see self.getHalfFlap)
            poolMats: vertex one-ring pooling matrix (see self.getFlapPool)
            dofs: degrees of freedom per vertex (see self.getFlapPool)
        """
        LC = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            HF = hfList[mIdx][0]
            poolMat = poolMats[mIdx][0]
            dof = dofs[mIdx][0]
            dV_he = V[HF[:, 0], :] - V[HF[:, 1], :]
            dV_v = torch.spmm(poolMat, dV_he)
            dV_v /= dof.unsqueeze(1)
            LC[mIdx] = dV_v
        return LC

    def computeParameters(self):
        """
        pre-compute parameters required for network training. It includes:
        hfList: list of half flaps
        poolMats: vertex one-ring pooling 
        LCs: vector of differential coordinates
        """
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)

    def old_toDevice(self, device):
        """
        move information to CPU/GPU
        """
        for ii in range(self.nM):
            for jj in range(self.nS):
                self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS - 1):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)

    def toDevice(self, device):
        """
        move information to CPU/GPU
        """
        for ii in range(self.nM):
            for jj in range(self.nS):
                self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)

    def toDeviceId(self, mIdx, device):
        """
        move information to CPU/GPU
        """
        for jj in range(self.nS):
            self.meshes[mIdx][jj].V = self.meshes[mIdx][jj].V.to(device)
            self.meshes[mIdx][jj].F = self.meshes[mIdx][jj].F.to(device)
        self.LCs[mIdx] = self.LCs[mIdx].to(device)
        for jj in range(self.nS):
            self.hfList[mIdx][jj] = self.hfList[mIdx][jj].to(device)
            self.poolMats[mIdx][jj] = self.poolMats[mIdx][jj].to(device)
            self.dofs[mIdx][jj] = self.dofs[mIdx][jj].to(device)

def preprocessTestShapes(meshPathList, nSubd=2):
    """
    process testing shapes given a list of .obj paths, including normalizing the shape and computing the half flap list
    """
    nObjs = len(meshPathList)
    meshes = [None] * nObjs
    scales = [None] * nObjs  # zzw
    for meshIdx in range(nObjs):
        path = meshPathList[meshIdx]
        V, F = tgp.readOBJ(path)
        # V = tgp.normalizeUnitCube(V) * 2  # zzw
        V, scale = tgp.my_normalizeUnitCube(V)  # zzw
        scales[meshIdx] = scale  # zzw
        FList, hfList = computeFlapList(V, F, nSubd)

        meshes_i = [None] * (nSubd + 1)
        for jj in range(nSubd + 1):
            if jj == 0:
                meshes_i[jj] = Mesh(V, FList[jj], hfList[jj])
            else:
                meshes_i[jj] = Mesh(None, FList[jj], hfList[jj])
        meshes[meshIdx] = list(meshes_i)
    print('num Subd: %d, num meshes: %d' % (nSubd, nObjs))
    return meshes, scales


class TestMeshes:
    def __init__(self, meshPathList, nSubd=2):
        """
        Inputs:
            meshPathList: list of pathes to .obj files
        """
        nShape = len(meshPathList)
        self.meshes, self.scales = preprocessTestShapes(meshPathList, nSubd)
        self.nM = len(self.meshes)  # number of meshes
        self.nS = len(self.meshes[0])  # number of subdivision levels

        # parameters
        self.hfList = None  # half flap index information
        self.poolMats = None  # vertex one-ring pooling matrices
        self.dofs = None  # vertex degrees of freedom
        self.LCs = None  # vector of differential coordinates

    def getScale(self, mIdx):
        """
        get scale parameter of input data
        Inputs:
            mIdx: mesh index
        """
        return self.scales[mIdx]  # ([[min_x,min_y,min_z], scale_length])

    def removeMesh(self, mIdxList):
        """
        remove a mesh from the list
        Inputs:
            mIdx: mesh index
        """
        for mIdx in mIdxList:
            self.meshes[mIdx] = None
            self.hfList[mIdx] = None
            self.poolMats[mIdx] = None
            self.dofs[mIdx] = None
            self.LCs[mIdx] = None
        self.meshes = [x for x in self.meshes if x is not None]
        self.hfList = [x for x in self.hfList if x is not None]
        self.poolMats = [x for x in self.poolMats if x is not None]
        self.dofs = [x for x in self.dofs if x is not None]
        self.LCs = [x for x in self.LCs if x is not None]

        self.nM = len(self.meshes)

    def getInputData(self, mIdx):
        """
        get input data for the network
        Inputs: 
            mIdx: mesh index
        """
        input = torch.cat((
            self.meshes[mIdx][0].V,
            self.LCs[mIdx]),
            dim=1)
        return input  # (nV x Din)

    def old_getHalfFlap(self):
        """
        create a list of half flap information, such that (see paper for the color scheme)
        HF[meshIdx][subdIdx] = [v_blue, v_red, v_purple, v_yellow]
        """
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS - 1)
            for jj in range(self.nS - 1):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getHalfFlap(self):
        """
        create a list of half flap information, such that (see paper for the color scheme)
        HF[meshIdx][subdIdx] = [v_blue, v_red, v_purple, v_yellow]
        """
        HF = [None] * self.nM
        for ii in range(self.nM):
            fifj = [None] * (self.nS)
            for jj in range(self.nS):
                idx = self.meshes[ii][jj].hfList[:, [0, 1, 2, 3]]
                fifj[jj] = idx.reshape(-1, 4)
            HF[ii] = list(fifj)
        return HF

    def getVertexParents(self, HF):
        """
        get the parent vertex of each vertex
        Inputs:
            HF: half flap list (see self.getHalfFlap())
        """
        nM = len(HF)
        nS = len(HF[0])
        vertexParents = [None] * nM
        for ii in range(nM):
            vertexParents_i = [None] * nS
            for jj in range(nS):
                vertexParents_j = [None] * 4
                hfIdx = HF[ii][jj]
                nV = hfIdx[:, 0].max() + 1

    def getFlapPool(self, HF):
        """
        get the matrix for vertex one-ring average pooling (left two sub-figures in Fig.17)
        Inputs:
            HF: half flap list (see self.getHalfFlap())
        """
        nM = len(HF)
        nS = len(HF[0])

        poolFlap = [None] * nM
        dof = [None] * nM
        for ii in range(nM):
            poolFlap_ij = [None] * nS
            dof_ij = [None] * nS
            for jj in range(nS):
                hfIdx = HF[ii][jj]
                nV = hfIdx[:, 0].max() + 1  # 顶点的最大索引号+1=顶点数。由于有些边无法计算half flap，所以这里可能点数有问题

                rIdx = hfIdx[:, 0]  # 2nE,最大值为nV-1
                cIdx = torch.arange(hfIdx.size(0))  # 2nE,[0,1,2,...,2nE-1]
                I = torch.cat([rIdx, cIdx], 0).reshape(2, -1)  # 2 * 2nE
                val = torch.ones(hfIdx.size(0))  # 2nE
                poolMat = torch.sparse.FloatTensor(I, val,
                                                   torch.Size([nV, hfIdx.size(0)]))  # nV * 2nE，记录了每个顶点的出现在第1位的边的索引
                # 原理是hfIdx记录了每条边正反两方向的按序排列的顶点索引号，所以每个点总会在边的正或反方向时位于4个顶点的第一位，所以顶点出现在第1位的次数，就代表了顶点相关联的边的数目。
                rowSum = torch.sparse.sum(poolMat,
                                          dim=1).to_dense()  # nV,每个顶点的度，相关联的边的数量也即相邻顶点的数量，度为0的点为孤立点。度为奇数的点为奇点，度为偶数的点为偶点

                poolFlap_ij[jj] = poolMat
                dof_ij[jj] = rowSum
            poolFlap[ii] = list(poolFlap_ij)
            dof[ii] = list(dof_ij)
        return poolFlap, dof

    def getLaplaceCoordinate(self, hfList, poolMats, dofs):
        """
        get the vectors of the differential coordinates (see Fig.18)
        Inputs:
            hfList: half flap list (see self.getHalfFlap)
            poolMats: vertex one-ring pooling matrix (see self.getFlapPool)
            dofs: degrees of freedom per vertex (see self.getFlapPool)
        """
        LC = [None] * self.nM
        for mIdx in range(self.nM):
            V = self.meshes[mIdx][0].V
            HF = hfList[mIdx][0]
            poolMat = poolMats[mIdx][0]
            dof = dofs[mIdx][0]
            dV_he = V[HF[:, 0], :] - V[HF[:, 1], :]
            dV_v = torch.spmm(poolMat, dV_he)
            dV_v /= dof.unsqueeze(1)
            LC[mIdx] = dV_v
        return LC

    def computeParameters(self):
        """
        pre-compute parameters required for network training. It includes:
        hfList: list of half flaps
        poolMats: vertex one-ring pooling 
        LCs: vector of differential coordinates
        """
        self.hfList = self.getHalfFlap()
        self.poolMats, self.dofs = self.getFlapPool(self.hfList)
        self.LCs = self.getLaplaceCoordinate(self.hfList, self.poolMats, self.dofs)

    def old_toDevice(self, device):
        """
        move information to CPU/GPU
        """
        for ii in range(self.nM):
            for jj in range(self.nS):
                if jj == 0:
                    self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS - 1):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)

    def toDevice(self, device):
        """
        move information to CPU/GPU
        """
        for ii in range(self.nM):
            for jj in range(self.nS):
                if jj == 0:
                    self.meshes[ii][jj].V = self.meshes[ii][jj].V.to(device)
                self.meshes[ii][jj].F = self.meshes[ii][jj].F.to(device)
        for ii in range(self.nM):
            self.LCs[ii] = self.LCs[ii].to(device)
            for jj in range(self.nS):
                self.hfList[ii][jj] = self.hfList[ii][jj].to(device)
                self.poolMats[ii][jj] = self.poolMats[ii][jj].to(device)
                self.dofs[ii][jj] = self.dofs[ii][jj].to(device)

    def toDeviceId(self, mIdx, device):
        """
        move information to CPU/GPU
        """
        for jj in range(self.nS):
            self.meshes[mIdx][jj].V = self.meshes[mIdx][jj].V.to(device)
            self.meshes[mIdx][jj].F = self.meshes[mIdx][jj].F.to(device)
        self.LCs[mIdx] = self.LCs[mIdx].to(device)
        for jj in range(self.nS):
            self.hfList[mIdx][jj] = self.hfList[mIdx][jj].to(device)
            self.poolMats[mIdx][jj] = self.poolMats[mIdx][jj].to(device)
            self.dofs[mIdx][jj] = self.dofs[mIdx][jj].to(device)
def zzw_computeFlapList(V, F, numSubd=2):
    """
    Compute lists of vertex indices for half flaps and for all subsequent subdivision levels. Each half flap has vertices ordered like:
    [v_blue, v_red, v_purple, v_yellow, v_blue_at_next_level]

    Inputs:
        V: nV-3 vertex list
        F: nF-3 face list
        numSubd: number of subdivisions
    """
    FList = []
    halfFlapList = []

    for iter in range(numSubd):  # 每个细分level的都要计算half Flap list
        # compute the subdivided vertex and face lists
        nV = V.size(0)  # 老顶点数量
        VV, FF, S = tgp_midPointUp(V, F, 1)  # 中点插入新顶点上采样，返回的是老+新顶点list，新的面list，以及记录了上采样后的老新顶点是由哪些老顶点计算得来的稀疏矩阵
        rIdx = S._indices()[0, :]  # 行索引，老+新顶点
        cIdx = S._indices()[1, :]  # 列索引，老顶点
        val = S._values()  # [1,,,,1,0.5,,,0.5]

        # only extract new vertices
        # Note: I order the vertex list as V = [oldV, newV]
        cIdx = cIdx[rIdx >= nV]
        val = val[rIdx >= nV]
        rIdx = rIdx[rIdx >= nV]
        assert ((val == 0.5).all())  # 上面把新顶点提取出来了。因为新顶点是两个老顶点的坐标，各0.5的权重插值来的

        rIdx, idx = torch.sort(rIdx)  # 将新顶点的索引从小到大排列，idx记录了新list中的元素在老list中的下标，排完大概这种效果[12,12,13,13,,]
        cIdx = cIdx[idx]  # 根据行的顺序变化，同步更新列的变化
        rIdx = rIdx[::2]  # 得到的是新顶点的索引list，维度是新顶点的个数，应该等于边的数量？rIdx[::2]=rIdx[0::2]，从索引0开始，以2为步长取值得到一个新的列表，下一个值的索引就是0+2=2
        cIdx = cIdx.view(-1, 2)  # cIdx变为两列，两个元素组成一行，1行记录的索引就是新顶点所在边的2个老顶点，-1表示计算得到行数
        # Note: Vodd = (V[cIdx[:,0],:] + V[cIdx[:,1],:]) / 2.0

        flapIdx = torch.zeros(len(rIdx), 4).long()  # 计算每条边的half flap，有4个顶点

        for kk in range(len(rIdx)):
            vi = cIdx[kk, 0]
            vj = cIdx[kk, 1]  # vi和vj是1条边对应的两个顶点的索引
            adjFi, _ = tgp.findIdx(F, vi)  # 找顶点i所在的面
            adjFj, _ = tgp.findIdx(F, vj)  # 找顶点j所在的面
            adjF = tgp.intersect1d(adjFi, adjFj)  # i,j同时在的面，就是half flap的两个flap面
            # assert (adjF.size(0) == 2)  # 这里报错，就是因为这条边只在一个面上，也就是说初始mesh网格不密闭，有不连通的三角形面片
            if adjF.size(0) == 2:
                f1 = F[adjF[0], :]  # 面1
                f2 = F[adjF[1], :]  # 面2

                # roll the index so that f1_vi[0] == f2_vi[0] == vi
                f1roll = -tgp.findIdx(f1, vi)[0]
                f2roll = -tgp.findIdx(f2, vi)[0]
                f1_vi = tgp.roll1d(f1, f1roll)
                f2_vi = tgp.roll1d(f2, f2roll)
                assert (f1_vi[0] == vi)
                assert (f2_vi[0] == vi)

                # check which one is f = [vi, vj, ?]
                if f1_vi[1] == vj:
                    f_first = f1_vi  # f_first = [vi, vj, ?]
                    f_sec = f2_vi  # f_sec   = [vi, ?, vj]
                elif f2_vi[1] == vj:
                    f_first = f2_vi  # f_first = [vi, vj, ?]
                    f_sec = f1_vi  # f_sec   = [vi, ?, vj]
                assert (f_first[1] == vj)
                assert (f_sec[2] == vj)

                # assemble flapIdx as
                # [v_blue, v_red, v_purple, v_yellow]
                flapIdx[kk, :] = torch.tensor([vi, vj, f_first[2], f_sec[1]])

            elif adjF.size(0) == 1:
                # 边界边， [v_i, v_j, v_i_prev, vj_next]
                # adjEi, _ = tgp.findIdx(cIdx, vi)  # 找顶点i所在的边
                # adjEj, _ = tgp.findIdx(cIdx, vj)  # 找顶点j所在的边

                current_f = F[adjF[0], :]
                # 初始化一个空的tensor
                vk = torch.empty(1)
                for i in current_f:
                    if bool(i != vi) & bool(i != vj):
                        vk = i
                        break
                flapIdx[kk, :] = torch.tensor([vi, vj, vk, vk])

        halfFlapIdx = flapIdx[:, [0, 1, 2, 3, 1, 0, 3, 2]]
        halfFlapIdx = halfFlapIdx.reshape(-1, 4)  # 2*nE，即两倍边的数量

        FList.append(F)
        halfFlapList.append(halfFlapIdx)

        V = VV
        F = FF

    FList.append(F)
    halfFlapList.append(None)
    return FList, halfFlapList


def computeFlapList(V, F, numSubd=2):
    """
    Compute lists of vertex indices for half flaps and for all subsequent subdivision levels. Each half flap has vertices ordered like:
    [v_blue, v_red, v_purple, v_yellow, v_blue_at_next_level]

    Inputs:
        V: nV-3 vertex list
        F: nF-3 face list
        numSubd: number of subdivisions
    """

    FList = []
    halfFlapList = []

    for iter in range(numSubd):
        # compute the subdivided vertex and face lists
        nV = V.size(0)
        VV, FF, S = tgp_midPointUp(V, F, 1)
        rIdx = S._indices()[0, :]
        cIdx = S._indices()[1, :]
        val = S._values()

        # only extract new vertices 
        # Note: I order the vertex list as V = [oldV, newV]
        cIdx = cIdx[rIdx >= nV]
        val = val[rIdx >= nV]
        rIdx = rIdx[rIdx >= nV]
        assert ((val == 0.5).all())

        rIdx, idx = torch.sort(rIdx)
        cIdx = cIdx[idx]
        rIdx = rIdx[::2]
        cIdx = cIdx.view(-1, 2)
        # Note: Vodd = (V[cIdx[:,0],:] + V[cIdx[:,1],:]) / 2.0

        flapIdx = torch.zeros(len(rIdx), 4).long()
        for kk in range(len(rIdx)):
            vi = cIdx[kk, 0]
            vj = cIdx[kk, 1]
            adjFi, _ = tgp.findIdx(F, vi)
            adjFj, _ = tgp.findIdx(F, vj)
            adjF = tgp.intersect1d(adjFi, adjFj)
            try:
                assert (adjF.size(0) == 2)
            except:
                return None, None

            f1 = F[adjF[0], :]
            f2 = F[adjF[1], :]

            # roll the index so that f1_vi[0] == f2_vi[0] == vi
            f1roll = -tgp.findIdx(f1, vi)[0]
            f2roll = -tgp.findIdx(f2, vi)[0]
            # 面其实是有方向的，这里的roll是为了让vi在第一个位置，这样改变顶点顺序不会影响面的方向
            # 比如（1，2，3）和（1，3，2）是不一样的
            # 但（1，2，3）和（3，2，1）是一样的
            f1_vi = tgp.roll1d(f1, f1roll)
            f2_vi = tgp.roll1d(f2, f2roll)
            assert (f1_vi[0] == vi)
            assert (f2_vi[0] == vi)

            # check which one is f = [vi, vj, ?]
            if f1_vi[1] == vj:
                f_first = f1_vi  # f_first = [vi, vj, ?]
                f_sec = f2_vi  # f_sec   = [vi, ?, vj]
            elif f2_vi[1] == vj:
                f_first = f2_vi  # f_first = [vi, vj, ?]
                f_sec = f1_vi  # f_sec   = [vi, ?, vj]
            assert (f_first[1] == vj)
            assert (f_sec[2] == vj)

            # assemble flapIdx as
            # [v_blue, v_red, v_purple, v_yellow]
            flapIdx[kk, :] = torch.tensor([vi, vj, f_first[2], f_sec[1]])

        # turn flap indices into half flap indices
        # note:
        # flapIdx =
        # [v_blue, v_red, v_purple, v_yellow]
        #    |
        #    V
        # halfFlapIdx = 
        # [v_blue, v_red, v_purple, v_yellow]
        # [v_red, v_blue, v_yellow, v_purple]
        #    |
        #    V
        # [v_blue, v_red, v_purple, v_yellow]
        # [v_blue, v_red, v_purple, v_yellow] (different orientation)
        halfFlapIdx = flapIdx[:, [0, 1, 2, 3, 1, 0, 3, 2]]
        halfFlapIdx = halfFlapIdx.reshape(-1, 4)

        FList.append(F)
        halfFlapList.append(halfFlapIdx)

        V = VV
        F = FF

    FList.append(F)
    halfFlapList.append(None)
    return FList, halfFlapList


def tgp_midPointUp(V, F, subdIter=1):
    """
    perform mid point upsampling
    """
    Vnp = V.data.numpy()
    Fnp = F.data.numpy()
    VVnp, FFnp, SSnp = midPointUpsampling(Vnp, Fnp, subdIter)
    VV = torch.from_numpy(VVnp).float()
    FF = torch.from_numpy(FFnp).long()

    SSnp = SSnp.tocoo()
    values = SSnp.data
    indices = np.vstack((SSnp.row, SSnp.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = SSnp.shape
    SS = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return VV, FF, SS


def random3DRotation():
    """
    generate a random 3D rotation matrix just for testing 
    """
    theta_x = torch.rand(1) * 2 * np.pi
    sinx = torch.sin(theta_x)
    cosx = torch.cos(theta_x)
    Rx = torch.tensor([[1., 0., 0.],
                       [0., cosx, -sinx],
                       [0., sinx, cosx]])

    theta_y = torch.rand(1) * 2 * np.pi
    siny = torch.sin(theta_y)
    cosy = torch.cos(theta_y)
    Ry = torch.tensor([[cosy, 0., siny],
                       [0., 1., 0.],
                       [-siny, 0., cosy]])

    theta_z = torch.rand(1) * 2 * np.pi
    sinz = torch.sin(theta_z)
    cosz = torch.cos(theta_z)
    Rz = torch.tensor([[cosz, -sinz, 0.],
                       [sinz, cosz, 0.],
                       [0., 0., 1.]])
    return Rx.mm(Ry).mm(Rz)
