# ข้อมูล
x = [29, 28, 34, 31, 25]
y = [77, 62, 93, 84, 59]

total_data = 5

sum_x = sum(x)
sum_y = sum(y)
x_bar = sum(x) / total_data
y_bar = sum(y) / total_data

Sxx = sum((xi - x_bar) ** 2 for xi in x)
Sxy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))

a = Sxy / Sxx
b = y_bar - a * x_bar

print(a)
print(b)

