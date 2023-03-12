import torch


def intersect1d(tensor1, tensor2):
    '''
    intersect1d return intersected elements between tensor1 and tensor2
    '''
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]  # 排序过后，相同索引号的值相邻
    return aux[:-1][(aux[:-1] == aux[1:]).data]  # aux[:-1]和aux[1:]和错位比较，结果是相邻元素比较，第0位和第1位比，，倒数第2位和最后1位比
    #  在比较结果为True的位置取值最后得到公共面的索引号
