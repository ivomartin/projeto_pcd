import random

with open('dados.csv', 'a') as file:
    for _ in range(100000):
        random_number = round(random.uniform(1, 100), 2)
        file.write(f'{random_number}\n')