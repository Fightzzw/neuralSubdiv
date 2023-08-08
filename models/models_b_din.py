import sys

sys.path.append('../')
from include import *


class MLP(torch.nn.Module):
    # This is the MLP template for the Initialization, Vertex, Edge networks (see Table 2 in the appendix)
    def __init__(self, Din, Dhid, Dout):
        '''
        Din: input dimension
        Dhid: a list of hidden layer size
        Dout: output dimension
        '''
        super(MLP, self).__init__()

        self.layerIn = torch.nn.Linear(Din, Dhid[0])
        self.hidden = torch.nn.ModuleList()
        for ii in range(len(Dhid) - 1):
            self.hidden.append(torch.nn.Linear(Dhid[ii], Dhid[ii + 1]))
        self.layerOut = torch.nn.Linear(Dhid[-1], Dout)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layerIn(x)
        x = self.relu(x)
        for ii in range(len(self.hidden)):
            x = self.hidden[ii](x)
            x = self.relu(x)
        x = self.layerOut(x)
        return x


class SubdNet(torch.nn.Module):
    # Subdivision network
    # This network consist of three MLPs (net_init, net_edge, net_vertex), and the forward pass is describe in the Section 5 of the paper
    def __init__(self, params):
        super(SubdNet, self).__init__()
        Din = params['Din']  # input dimension
        Dout = params['Dout']  # output dimension

        # initialize three MLPs
        self.net_init = MLP(4 * Din + 6, params['h_initNet'], Dout)  # 4 * Din - 3 + 5 + 4
        self.net_edge = MLP(4 * Dout -3, params['h_edgeNet'], Dout)
        self.net_vertex = MLP(4 * Dout -3, params['h_vertexNet'], Dout)

        self.pool = torch.nn.AvgPool2d((2, 1))  # half-edge pool
        self.numSubd = params["numSubd"]  # number of subdivisions

    def flapNormalization(self, hf, normalizeFeature=False):
        """
        FLAPNORMALIZATION normalize the features of a half flap so that it is orientation and translation invariant (see Section 5)

        inputs:
          hf: 2*nE x 4 x Dim tensor of half flap features (in world coordinates)
          normalizeFeature: True/False whether to normalize the feature vectors

        output:
          hf_normalize: 2*nE x 4 x Dim tensor of half flap features (in local coordinates)
          localFrames a 3-by-3 matrix [b1; b2; b3] with frames b1, b2, b3

        Note:
        we only set "normalizeFeature" to True in the initialization network to make the differential coordinate features invariant to rigid motions, see figure 18 (top)
        """

        V = hf[:, :, :3]  # half flap vertex positison
        F = torch.tensor([[0, 1, 2], [1, 0, 3]])  # half flap face list

        # 1st frame: edge vector
        b1 = (V[:, 1, :] - V[:, 0, :]) / torch.norm(V[:, 1, :] - V[:, 0, :], dim=1).unsqueeze(1)

        # 3rd frame: edge normal (avg of face normals)
        vec1 = V[:, F[:, 1], :] - V[:, F[:, 0], :]
        vec2 = V[:, F[:, 2], :] - V[:, F[:, 0], :]
        FN = torch.cross(vec1, vec2)  # nF x 2 x 3
        FNnorm = torch.norm(FN, dim=2)
        FN = FN / FNnorm.unsqueeze(2)

        eN = FN[:, 0, :] + FN[:, 1, :]

        # zzw add
        # zero_rows = (eN == 0).all(dim=1)
        # # 将eN中整行元素为0的行用FN[:,0,:]的对应行替换
        # eN[torch.where(zero_rows)] = FN[:,0,:][torch.where(zero_rows)]

        b3 = eN / torch.norm(eN, dim=1).unsqueeze(1)

        # 2nd frame: their cross product
        b2 = torch.cross(b3, b1)

        # concatenage all local frames
        b1 = b1.unsqueeze(1)
        b2 = b2.unsqueeze(1)
        b3 = b3.unsqueeze(1)
        localFrames = torch.cat((b1, b2, b3), dim=1)

        # normalize features
        hf_pos = hf[:, :, :3]  # half flap vertex position
        hf_feature = hf[:, :, 3:]  # half flap features
        hf_pos = hf_pos - V[:, 0, :].unsqueeze(1)  # translate
        hf_pos = torch.bmm(hf_pos, torch.transpose(localFrames, 1, 2))
        if normalizeFeature:  # if also normalize the feature using local frames
            assert (hf_feature.size(2) == 3)
            # 对拉普拉斯坐标进行normalize
            hf_feature = torch.bmm(hf_feature, torch.transpose(localFrames, 1, 2))
        hf_normalize = torch.cat((hf_pos, hf_feature), dim=2)

        return hf_normalize, localFrames

    def v2hf(self, fv, hfIdx):
        '''
        V2HF re-index the vertex feature (fv) to half flaps features (hf), given half flap index list (hfIdx)
        '''
        # get half flap indices
        fv0 = fv[hfIdx[:, 0], :].unsqueeze(1)  # 2*nE x 1 x Dout
        fv1 = fv[hfIdx[:, 1], :].unsqueeze(1)  # 2*nE x 1 x Dout
        fv2 = fv[hfIdx[:, 2], :].unsqueeze(1)  # 2*nE x 1 x Dout
        fv3 = fv[hfIdx[:, 3], :].unsqueeze(1)  # 2*nE x 1 x Dout
        hf = torch.cat((fv0, fv1, fv2, fv3), dim=1)  # 2*nE x 4 x Dout

        # normalize the half flap features
        hf_normalize, localFrames = self.flapNormalization(hf)
        hf_normalize = hf_normalize.view(hf_normalize.size(0), -1)
        hf_normalize = hf_normalize[:, 3:]  # remove the first 3 components as they are always (0,0,0)

        return hf_normalize, localFrames

    def v2hf_initNet(self, fv, hfIdx, poolMat):
        '''
        V2HF_INITNET re-index the vertex feature (fv) to half flaps features (hf), given half flap index list (hfIdx). This is for the initialization network only
        '''
        # get half flap indices
        fv0 = fv[hfIdx[:, 0], :].unsqueeze(1)  # 2*nE x 1 x Dout
        fv1 = fv[hfIdx[:, 1], :].unsqueeze(1)  # 2*nE x 1 x Dout
        fv2 = fv[hfIdx[:, 2], :].unsqueeze(1)  # 2*nE x 1 x Dout
        fv3 = fv[hfIdx[:, 3], :].unsqueeze(1)  # 2*nE x 1 x Dout
        hf = torch.cat((fv0, fv1, fv2, fv3), dim=1)  # 2*nE x 4 x Dout

        # normalize the half flap features (including the vector of differential coordinates see figure 18)
        hf_normalize, localFrames = self.flapNormalization(hf, True)

        hf_normalize = hf_normalize.view(hf_normalize.size(0), -1)
        hf_normalize = hf_normalize[:, 3:]  # remove the first 3 components as they are always (0,0,0)

        newFeature =True

        if newFeature:
            V = hf[:, :, :3]  # half flap vertex positison
            # zzw add new features
            # 二面角，两个内角，两个底边/高，其中角度用余弦值表示
            # TODO : 判断二面角是a还是（2pi-a)
            F = torch.tensor([[0, 1, 2], [1, 0, 3]])  # half flap face list
            # 两个面的法向量
            vec1 = V[:, F[:, 1], :] - V[:, F[:, 0], :]
            vec2 = V[:, F[:, 2], :] - V[:, F[:, 0], :]
            FN = torch.cross(vec1, vec2)  # nF x 2 x 3
            FNnorm = torch.norm(FN, dim=2)
            FN = FN / FNnorm.unsqueeze(2)
            dihedral_angle = torch.sum(FN[:, 0, :] * FN[:, 1, :], dim=1)  # 二面角，因为FN已经归一化了，所以这里直接相乘
            # 两个底边/高
            vec_02 = V[:, 2, :] - V[:, 0, :]
            vec_03 = V[:, 3, :] - V[:, 0, :]
            vec_01 = V[:, 1, :] - V[:, 0, :]
            len_vec_01 = torch.norm(vec_01, dim=1)
            vec_01x02 = torch.cross(vec_01, vec_02)
            vec_01x03 = torch.cross(vec_01, vec_03)
            edge_ratio_2 = torch.norm(vec_01x02, dim=1) / (len_vec_01 ** 2)
            edge_ratio_3 = torch.norm(vec_01x03, dim=1) / (len_vec_01 ** 2)

            # 两个内角
            vec_20 = V[:, 0, :] - V[:, 2, :]
            vec_21 = V[:, 1, :] - V[:, 2, :]
            vec_30 = V[:, 0, :] - V[:, 3, :]
            vec_31 = V[:, 1, :] - V[:, 3, :]

            angle_2 = torch.sum(vec_20 * vec_21, dim=1) / (torch.norm(vec_20, dim=1) * torch.norm(vec_21, dim=1))
            angle_3 = torch.sum(vec_30 * vec_31, dim=1) / (torch.norm(vec_30, dim=1) * torch.norm(vec_31, dim=1))

            # half flap 4个顶点的高斯曲率
            # 每个顶点的高斯曲率分别计算
            angel_201 = torch.arccos(
                torch.sum(vec_02 * vec_01, dim=1) / (torch.norm(vec_02, dim=1) * torch.norm(vec_01, dim=1)))
            angel_301 = torch.arccos(
                torch.sum(vec_03 * vec_01, dim=1) / (torch.norm(vec_03, dim=1) * torch.norm(vec_01, dim=1)))
            hf_angle = (angel_201 + angel_301).unsqueeze(1)
            hf_area = torch.norm(vec_01x02, dim=1) + torch.norm(vec_01x03, dim=1)
            hf_area = hf_area.unsqueeze(1)

            # 顶点的高斯曲率
            aver_hf_angle = torch.spmm(poolMat, hf_angle)/2
            aver_hf_area = torch.spmm(poolMat, hf_area)/2
            v_gaussian_curvature = (2*np.pi - aver_hf_angle) / aver_hf_area
            hf_gaussian_curvature = torch.cat((v_gaussian_curvature[hfIdx[:, 0], :],
                                               v_gaussian_curvature[hfIdx[:, 1], :],
                                               v_gaussian_curvature[hfIdx[:, 2], :],
                                               v_gaussian_curvature[hfIdx[:, 3], :]), dim=1)

            mesh_cnn_feat = torch.cat((dihedral_angle.unsqueeze(1),
                                       angle_2.unsqueeze(1),
                                       angle_3.unsqueeze(1),
                                       edge_ratio_2.unsqueeze(1),
                                       edge_ratio_3.unsqueeze(1)), dim=1)
            hf_normalize = torch.cat((hf_normalize, mesh_cnn_feat, hf_gaussian_curvature), dim=1)

        return hf_normalize, localFrames

    def local2Global(self, hf_local, LFs):
        '''
        LOCAL2GLOBAL turns position features (the first three elements) described in the local frame of an half-flap to world coordinates
        '''
        hf_local_pos = hf_local[:, :3]  # get the vertex position features
        hf_feature = hf_local[:, 3:]  # get the high-dim features
        c0 = hf_local_pos[:, 0].unsqueeze(1)
        c1 = hf_local_pos[:, 1].unsqueeze(1)
        c2 = hf_local_pos[:, 2].unsqueeze(1)
        hf_global_pos = c0 * LFs[:, 0, :] + c1 * LFs[:, 1, :] + c2 * LFs[:, 2, :]
        hf_global = torch.cat((hf_global_pos, hf_feature), dim=1)
        return hf_global

    def halfEdgePool(self, fhe):
        '''
        average pooling of half edge features, see figure 17 (right)
        '''
        fhe = fhe.unsqueeze(0).unsqueeze(0)
        fe = self.pool(fhe)
        fe = fe.squeeze(0).squeeze(0)
        return fe

    def oneRingPool(self, fhe, poolMat, dof):
        '''
        average pooling over vertex one rings, see figure 17 (left, middle))
        '''
        fv = torch.spmm(poolMat, fhe)
        fv /= dof.unsqueeze(1)  # average pooling
        return fv

    def edgeMidPoint(self, fv, hfIdx):
        '''
        get the mid point position of each edge
        '''
        Ve0 = fv[hfIdx[:, 0], :3]
        Ve1 = fv[hfIdx[:, 1], :3]
        Ve = (Ve0 + Ve1) / 2.0
        Ve = self.halfEdgePool(Ve)
        return Ve

    def forward(self, fv, HF, poolMat, DOF):
        outputs = []
        vertex_parents = []
        # 将fv的元素的索引写入vertex_parents

        # initialization step (figure 17 left)
        fv_input_pos = fv[:, :3]
        fhf, LFs = self.v2hf_initNet(fv, HF[0], poolMat[0])
        fhf = self.net_init(fhf)
        fhf = self.local2Global(fhf, LFs)
        fv = self.oneRingPool(fhf, poolMat[0], DOF[0])
        fv[:, :3] += fv_input_pos

        outputs.append(fv[:, :3])

        # subdivision starts
        for ii in range(self.numSubd):
            # vertex step (figure 17 middle)
            prevPos = fv[:, :3]
            fhf, LFs = self.v2hf(fv, HF[ii])  # 2*nE x 4*Dout
            fhf = self.net_vertex(fhf)
            fhf = self.local2Global(fhf, LFs)
            fv = self.oneRingPool(fhf, poolMat[ii], DOF[ii])
            fv[:, :3] += prevPos
            fv_even = fv

            # edge step (figure 17 right)
            Ve = self.edgeMidPoint(fv, HF[ii])  # compute mid point
            fhf, LFs = self.v2hf(fv, HF[ii])  # 2*nE x 4*Dout
            fv_odd = self.net_edge(fhf)  # 2*nE x Dout
            fv_odd = self.local2Global(fv_odd, LFs)
            fv_odd = self.halfEdgePool(fv_odd)  # nE x Dout
            fv_odd[:, :3] += Ve

            # concatenate results
            fv = torch.cat((fv_even, fv_odd), dim=0)  # nV_next x Dout
            outputs.append(fv[:, :3])

        return outputs
