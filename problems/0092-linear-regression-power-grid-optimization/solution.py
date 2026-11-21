import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)
	sum_x = 0  															# add i
	sum_y = 0  															#  add detrended
	sum_xy = 0  														# add i * detrended
	sum_x2 = 0  														#  add i * i
	for i in range(1,len(consumption_data) + 1):
		fluctuation = (10 * math.sin((2*PI*i) / 10))
		detrended = consumption_data[i-1] - fluctuation
		sum_x += i
		sum_y += detrended
		sum_xy += i * detrended
		sum_x2 += i * i

	m = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_x2) - sum_x**2)
	b = (sum_y - m*sum_x) / n
	base_15 = m * 15 + b
	f_15 = 10 * math.sin(2 * PI * 15 / 10)
	pred_15 = base_15 + f_15
	final_15 = math.ceil(1.05 * round(pred_15))

	return final_15