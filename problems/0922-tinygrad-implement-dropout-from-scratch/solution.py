from tinygrad import Tensor

def dropout(x, p, training):
    # TODO: inverted dropout
    if not training or p == 0 :
        return x 
    
    noise = Tensor.rand(*x.shape)

    mask = (noise >= p).cast(x.dtype)

    return x * mask / (1 - p)
