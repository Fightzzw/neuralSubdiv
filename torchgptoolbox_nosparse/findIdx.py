import torch
def findIdx(F, VIdx):
    '''
    FINDIDX finds desired indices in a torch tensor

    Inputs:
    F: |F|-by-dim torch tensor 
    VIdx: a list of indices

    Output:
    r, c: row/colummn indices in the torch tensor
    '''

    def isin(ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)

    mask = isin(F.view(-1),VIdx)  # F.view(-1) 将F变为1维的了，即长度=面数*3，为的是方便后面用torch.where()查找
    try:
        nDim = F.shape[1]
    except:
        nDim = 1  #面的维度，三角形面就是3
    r = torch.floor(torch.where(mask)[0] / (nDim*1.0) ).long()  #因为上面把F变成一维的了，找出mask的值为True的索引，除以3，得到面的索引
    c = torch.where(mask)[0] % nDim
    return r,c