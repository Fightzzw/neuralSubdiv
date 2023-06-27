import torch

from include import *
import trimesh
import numpy as np
import random

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
        self.net_init = MLP(4 * Din - 3, params['h_initNet'], Dout)
        self.net_edge = MLP(4 * Dout - 3, params['h_edgeNet'], Dout)
        self.net_vertex = MLP(4 * Dout - 3, params['h_vertexNet'], Dout)

        self.pool = torch.nn.AvgPool2d((2, 1))  # half-edge pool
        self.numSubd = params["numSubd"]  # number of subdivisions

    def get_min_sphere_radius(self, points):
        """
        Get the minimum radius of a circumscribed sphere that encloses all the points
        """
        # 外接球半径
        center, radius = trimesh.nsphere.minimum_nsphere(points)
        return center, radius

    def get_min_sphere_4points(self, points):
        """
        Get the minimum radius of a circumscribed sphere that encloses all the points
        """
        def minimum_enclosing_sphere_3points(triangle):
            # Compute the circumcenter of the triangle
            a, b, c = triangle
            ab = b - a
            ac = c - a
            ab_cross_ac = np.cross(ab, ac)
            ab_cross_ac_norm_sq = np.dot(ab_cross_ac, ab_cross_ac)
            if ab_cross_ac_norm_sq == 0:
                # Points are colinear, return a point and radius of infinity
                return a, np.inf
            ab_norm_sq = np.dot(ab, ab)
            ac_norm_sq = np.dot(ac, ac)
            circumcenter = a + (np.cross(ab_norm_sq * ac - ac_norm_sq * ab, ab_cross_ac) / (2 * ab_cross_ac_norm_sq))
            # Calculate the radius of the circumcircle
            radius = np.linalg.norm(circumcenter - a)
            # Check if the circumcenter lies inside the triangle
            if np.all(np.logical_and(circumcenter >= a, circumcenter <= c)):
                return circumcenter, radius
            # Otherwise, the minimum enclosing sphere is the circumcircle
            else:
                center = np.mean(triangle, axis=0)
                radius = np.max(np.linalg.norm(triangle - center, axis=1))
                return center, radius
        def _min_sphere(points, center, radius):
            if len(points) == 0 or len(center) == 3:
                if len(center) == 3:
                    # c1, c2, c3 = center
                    # return np.array([(c1 + c2 + c3) / 3]), 0
                    return minimum_enclosing_sphere_3points(center)
                elif len(center) == 2:
                    c1, c2 = center
                    return (c1 + c2) / 2, np.linalg.norm(c1 - c2) / 2
                elif len(center) == 1:
                    return center[0], 0
                else:
                    return None, 0
            else:
                p = points[0]
                points = points[1:]
                c, r = _min_sphere(points, center, radius)
                if c is None or np.linalg.norm(p - c) > r:
                    center.append(p)
                    c, r = _min_sphere(points, center, radius)
                    center.pop()
                return c, r

        if len(points) < 4:
            raise ValueError("At least 4 points are required.")
        np.random.shuffle(points)
        center, radius = _min_sphere(points, [], 0)
        return center, radius

    def fit_sphere_4points(self, array, tol=1e-6):

        # Check if the the points are co-linear
        D12 = array[1] - array[0]
        D12 = D12 / np.linalg.norm(D12)
        D13 = array[2] - array[0]
        D13 = D13 / np.linalg.norm(D13)
        D14 = array[3] - array[0]
        D14 = D14 / np.linalg.norm(D14)

        chk1 = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)
        chk2 = np.clip(np.abs(np.dot(D12, D14)), 0., 1.)
        if np.arccos(chk1) / np.pi * 180 < tol or np.arccos(chk2) / np.pi * 180 < tol:
            R = np.inf
            C = np.full(3, np.nan)
            return C, R

        # Check if the the points are co-planar
        n1 = np.linalg.norm(np.cross(D12, D13))
        n2 = np.linalg.norm(np.cross(D12, D14))

        chk = np.clip(np.abs(np.dot(n1, n2)), 0., 1.)
        if np.arccos(chk) / np.pi * 180 < tol:
            R = np.inf
            C = np.full(3, np.nan)
            return C, R

        # Centroid of the sphere
        A = 2 * (array[1:] - np.full(len(array) - 1, array[0]))
        b = np.sum((np.square(array[1:]) - np.square(np.full(len(array) - 1, array[0]))), axis=1)
        C = np.transpose(np.linalg.solve(A, b))

        # Radius of the sphere
        R = np.sqrt(np.sum(np.square(array[0] - C), axis=0))
        # print("Center:", C)
        # print("Radius:", R)

        return C, R

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
        # zzw add
        # 1. use the length of the edge vector12 to normalize the hf_pos
        # calculate the length of the edge vector12
        # normalize_length = torch.norm(hf_pos[:, 1, :], dim=1).unsqueeze(1).unsqueeze(1)

        # 2. normalize_length = radius of the circumscribed sphere of half flap
        # calculate the circumscribed sphere of the half flap
        # normalize_length = torch.ones(hf_pos.size(0))
        # max_normazlize_length = 100000
        # for i in range(hf_pos.size(0)):
        #     _, radius = self.fit_sphere_4points(hf_pos[i, :, :].cpu().detach().numpy())
        #     if radius == np.inf:
        #         print('%d/%d' % (i, hf_pos.size(0)))
        #         print(hf_pos[i, :, :].cpu().detach().numpy())
        #         print('radius is inf, we set it to max_normazlize_length = %d' % max_normazlize_length)
        #         radius = max_normazlize_length
        #     normalize_length[i] = torch.from_numpy(np.array(radius))
        #
        # normalize_length = normalize_length.unsqueeze(1).unsqueeze(1).cuda()
        # 3. normalize_length = len_edge12 * len_edge34
        # len_edge12 = torch.norm(hf_pos[:, 1, :], dim=1).unsqueeze(1).unsqueeze(1)
        # len_edge34 = torch.norm(hf_pos[:, 3, :] - hf_pos[:, 2, :], dim=1).unsqueeze(1).unsqueeze(1)
        # normalize_length = len_edge12 * len_edge34
        # # 对normalize_length开根号
        # normalize_length = torch.sqrt(normalize_length)

        # 4. normalize_length = radius of the min sphere of half flap
        normalize_length = torch.ones(hf_pos.size(0))
        for i in range(hf_pos.size(0)):
            _, radius = self.get_min_sphere_4points(hf_pos[i, :, :].cpu().detach().numpy())
            # print('%d/%d, radius: ' % (i, hf_pos.size(0)), radius)
            normalize_length[i] = torch.from_numpy(np.array(radius))
        normalize_length = normalize_length.unsqueeze(1).unsqueeze(1).cuda()

        hf_pos = hf_pos / normalize_length

        if normalizeFeature:  # if also normalize the feature using local frames
            assert (hf_feature.size(2) == 3)
            hf_feature = torch.bmm(hf_feature, torch.transpose(localFrames, 1, 2))
            hf_feature = hf_feature / normalize_length
        hf_normalize = torch.cat((hf_pos, hf_feature), dim=2)
        return hf_normalize, localFrames, normalize_length

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
        hf_normalize, localFrames, normalize_length = self.flapNormalization(hf)
        hf_normalize = hf_normalize.view(hf_normalize.size(0), -1)
        hf_normalize = hf_normalize[:, 3:]  # remove the first 3 components as they are always (0,0,0)
        return hf_normalize, localFrames, normalize_length

    def v2hf_initNet(self, fv, hfIdx):
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
        hf_normalize, localFrames, normalize_length = self.flapNormalization(hf, True)
        hf_normalize = hf_normalize.view(hf_normalize.size(0), -1)
        hf_normalize = hf_normalize[:, 3:]  # remove the first 3 components as they are always (0,0,0)
        return hf_normalize, localFrames, normalize_length

    def local2Global(self, hf_local, LFs, normalize_length):
        '''
        LOCAL2GLOBAL turns position features (the first three elements) described in the local frame of an half-flap to world coordinates  
        '''
        hf_local_pos = hf_local[:, :3] * normalize_length.squeeze(1)  # get the vertex position features
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

    def forward(self, fv, mIdx, HFs, poolMats, DOFs):
        outputs = []

        # initialization step (figure 17 left)
        fv_input_pos = fv[:, :3]
        fhf, LFs, normalize_length = self.v2hf_initNet(fv, HFs[mIdx][0])
        fhf = self.net_init(fhf)
        fhf = self.local2Global(fhf, LFs, normalize_length)
        fv = self.oneRingPool(fhf, poolMats[mIdx][0], DOFs[mIdx][0])
        fv[:, :3] += fv_input_pos

        outputs.append(fv[:, :3])

        # subdivision starts
        for ii in range(self.numSubd):
            # vertex step (figure 17 middle)
            prevPos = fv[:, :3]
            fhf, LFs, normalize_length = self.v2hf(fv, HFs[mIdx][ii])  # 2*nE x 4*Dout
            fhf = self.net_vertex(fhf)  # 2*nE x Dout
            fhf = self.local2Global(fhf, LFs, normalize_length)
            fv = self.oneRingPool(fhf, poolMats[mIdx][ii], DOFs[mIdx][ii])  # nv x Dout
            fv[:, :3] += prevPos
            fv_even = fv

            # edge step (figure 17 right)
            Ve = self.edgeMidPoint(fv, HFs[mIdx][ii])  # compute mid point
            fhf, LFs, normalize_length = self.v2hf(fv, HFs[mIdx][ii])  # 2*nE x 4*Dout
            fv_odd = self.net_edge(fhf)  # 2*nE x Dout
            fv_odd = self.local2Global(fv_odd, LFs, normalize_length)
            fv_odd = self.halfEdgePool(fv_odd)  # nE x Dout
            fv_odd[:, :3] += Ve

            # concatenate results
            fv = torch.cat((fv_even, fv_odd), dim=0)  # nV_next x Dout
            outputs.append(fv[:, :3])

        return outputs
