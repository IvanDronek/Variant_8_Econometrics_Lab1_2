# Множинна регресія
from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f"8-й варіант")

# print("Вводимо дані")
y = [33.18, 30.45, 37.77, 40.59, 41.07, 44.10, 43.65, 52.50, 52.92,
     60.60, 55.95, 57.02, 57.52, 59.60, 61.15, 62.54]

x = [
    [15.86, 3.48, 33.25], [14.57, 4.05, 43.15], [15.31, 5.03, 41.05], [14.85, 6.26, 40.65], [18.18, 6.20, 49.30],
    [19.64, 7.76, 56.10], [17.22, 9.06, 57.60], [18.99, 8.45, 51.10], [21.51, 9.88, 60.65], [23.70, 9.56, 61.95],
    [24.02, 12.82, 57.20], [24.56, 11.56, 59.35], [24.27, 11.58, 60.21], [26.32, 12.95, 61.25], [25.99, 12.35, 63.36],
    [27.00, 13.65, 64.02]
]
x, y = np.array(x), np.array(y)


# print("Створюємо модель:")
model = LinearRegression().fit(x, y)
print(model)


# print("Отримуємо результати:")
# R^2 = (score)
# round(), 4 округлення
r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅^2): {r_sq}")

print(f"intercept (a, 𝑏0): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏1): {model.coef_}")


# print("Прогнозуємо відповідь")
# g(xi) задяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) з формули
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

# модель для нових даних
x_new = np.arange(10).reshape((-1, 3))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.YELLOW + f"Висновки:"
                 f"\n (𝑅^2): {r_sq},"
                 f"\n (a, 𝑏0): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏1): {model.coef_}")


