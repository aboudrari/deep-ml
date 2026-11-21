def phi_corr(x: list[int], y: list[int]) -> float:
    x00 = 0
    x01 = 0
    x10 = 0
    x11 = 0
    for xi, yi in zip(x, y):
        if xi == 0 and yi == 0:
            x00 += 1
        elif xi == 0 and yi == 1:
            x01 += 1
        elif xi == 1 and yi == 0:
            x10 += 1
        else:
            x11 += 1
    val = ((x00 * x11) - (x01 * x10)) / ((x00+x01)*(x10+x11)*(x00+x10)*(x01+x11)) ** 0.5
    return round(val, 4)