#Лінійна регресія
from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f" 8-й варіант")

# print("Введемо дані")
print(f"Подамо x як двовимірний масив:")
x = np.array([6.7, 7.2, 8.1, 11.2, 12.6, 13.3, 15.6, 17.9, 18, 18.9]).reshape((-1, 1))
y = np.array([2.3, 3.1, 6.5, 7.6, 8.4, 9.7, 11.3, 12.8, 13.5, 14.6])



model = LinearRegression()
print(model.fit(x, y))


r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅^2): {r_sq}")

print(f"intercept (a, 𝑏0): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏1): {model.coef_}")

print(f"Подамо і y як двовимірний масив:")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept (a, 𝑏0): {new_model.intercept_}")
print(f"slope (b, 𝑏1): {new_model.coef_}")


# print("Прогнозуємо відповідь")
# g(xi) завдяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) з формули
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

# модель для нових даних
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.YELLOW + f"Висновки:"
                 f"\n (𝑅^2): {r_sq},"
                 f"\n (a, 𝑏0): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏1): {model.coef_}")