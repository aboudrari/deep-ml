import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    
    if p == 0 :
        return x
    if training  :

        r = torch.rand(x.shape)
        m = torch.where(r >= p, 1.0, 0.0)

        y = (x * m )/(1 - p)

        return y 

    else :
        return x 
            
