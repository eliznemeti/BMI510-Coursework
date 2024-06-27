''' 1. Do a simple linear regression. Assuming:
•	x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
•	y = [589.6, 1173.8, 1758, 2344.8, 2930, 3514.7, 4098.8, 4685, 5269, 5854]
•	Regress y onto x. Identify the slope and intercept.
•	Use any interpreted or compiled language. (No spreadsheets.) '''

import pandas as pd
import statsmodels.api as sm

#create DataFrame
df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'y': [589.6, 1173.8, 1758, 2344.8, 2930, 3514.7, 4098.8, 4685, 5269, 5854]})

#define response variable
y = df['y']

#define predictor variables
x = df['x']

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

''' 2.	Use a MacLaurin series to estimate π. Recalling:
•	arctan(x) = x – (1/3)•x3 + (1/5)•x5 – (1/7)•x7 + ...
•	arctan(1) = π / 4v
•	Write a function f(n) that returns an estimate of π based on n MacLaurin series terms.
•	Use any interpreted or compiled language. (No spreadsheets.)
•	4*arctan(1) = 4*(1 – (1/3)*1 + (1/5)*1 – (1/7)*1)'''


def pi_approx(n):
    sol = 0
    i = 1
    while i <= n:
        coeff = 2*i-1
        if i % 2 == 0:
            coeff *= -1
        sol += (1/coeff)
        i += 1
    sol *= 4
    return sol

print(pi_approx(20000))