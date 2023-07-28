import torch

def normalizeUnitCube(V):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 0.5,0.5,0.5

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    V = V - torch.min(V,0)[0].unsqueeze(0)
    # x_min = torch.min(V[:,0])
    # y_min = torch.min(V[:,1])
    # z_min = torch.min(V[:,2])
    # min_bound = torch.tensor([x_min, y_min, z_min]).unsqueeze(0)
    # V = V - min_bound

    V = V / torch.max(V.view(-1)) / 2.0
    return V

def my_normalizeUnitCube(V):
    '''
    NORMALIZEUNITCUBE normalize a shape to the bounding box by 1.0,1.0,1.0

    Inputs:
        V (|V|,3) torch array of vertex positions

    Outputs:
        V |V|-by-3 torch array of normalized vertex positions
    '''
    scale =[]
    scale.append(torch.min(V,0)[0])
    V = V - torch.min(V,0)[0].unsqueeze(0)
    scale.append(torch.max(V.view(-1)))
    V = V / torch.max(V.view(-1))
    return V, scale
