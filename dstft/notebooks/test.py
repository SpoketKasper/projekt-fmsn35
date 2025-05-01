import torch 

def generate_bspline(self):
    c = self.spline_stride
    bspline = torch.ones((c))
    length = c
    p = 0
    while p < self.spline_degree:
        p += 1
        length += c
        first = torch.zeros((length))
        first[:-c] = bspline
        second = torch.zeros((length))
        second[c:] = bspline
        bspline = 1/(p*c) * ( torch.arange(length)*first + 
                                   ((p+1)*c -torch.arange(length))*second )
    return bspline
