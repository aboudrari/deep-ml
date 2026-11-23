import  numpy as np  

def compute_partial_derivatives(func_name, point):
    if len(point) == 2:
        x, y = point
    else:
        x, y, z = point

    if func_name == 'poly2d':
        return (2*x*y + y**2, x**2 + 2*x*y)
    elif func_name == 'exp_sum':
        return (np.exp(x+y), np.exp(x+y))
    elif func_name == 'product_sin':
        return (np.sin(y), x*np.cos(y))
    elif func_name == 'poly3d':
        return (2*x*y, x**2 + z**2, 2*y*z)
    elif func_name == 'squared_error':
        return (2*(x-y), -2*(x-y))