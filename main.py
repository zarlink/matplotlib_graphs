# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import csv
import datetime as dt
import matplotlib.pyplot as plt


Date, Daily_Installs = [], []
with open('analisis_sueldos.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)

    for line in csv_reader:
        data = line[0].split(",")  # Separar la línea en fecha y número
        print(data[0])
        date = dt.datetime.strptime(data[0], '%d/%m/%Y')
        daily_installs = int(data[1])

        # Agregar fecha y número a los arreglos correspondientes
        Date.append(date)
        Daily_Installs.append(daily_installs)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Date,Daily_Installs,'o-')
fig.autofmt_xdate()

plt.show()