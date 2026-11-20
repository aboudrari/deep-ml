
def find_treasure(start_x: float) -> float:
    learning_rate = 0.01
    momentum = 0.9
    tolerance = 1e-8
    max_iters = 10000

    def gradient(x):
        return 4 * x**3 - 9 * x**2

    x = start_x
    velocity = 0.0

    for _ in range(max_iters):
        grad = gradient(x)

        velocity = momentum * velocity + learning_rate * grad
        new_x = x - velocity

        if abs(new_x - x) < tolerance and abs(grad) > 1e-8:
            pass

        x = new_x

        if abs(x - 2.25) < tolerance:
            break

    return round(x, 4)
