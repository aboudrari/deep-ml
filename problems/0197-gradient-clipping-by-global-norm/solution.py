import numpy as np 

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    
    gradients = [np.array(g, dtype=float) for g in gradients]

    sum_grad = sum(np.sum(g ** 2) for g in gradients)
    global_norm = np.sqrt(sum_grad)

    if global_norm > max_norm:

        scaling_fac = max_norm / global_norm
        result = [scaling_fac * g for g in gradients]
        return [r.tolist() for r in result]

    else:
        return [g.tolist() for g in gradients]