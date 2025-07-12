import pandas as pd

data = [
    [22, "SUV", 25, 4, 2.0, "AWD", "Gasolina", 30, "Honda", "CR-V", "Automática", 2020]
]

columnas = [
    "mpg_ciudad","clase","mpg_combinado","cilindros","cilindrada","conduccion",
    "tipo_combustible","mpg_carretera","marca","modelo","transmision","ano"
]

df = pd.DataFrame(data, columns=columnas)
df.to_csv("sample.csv", index=False)