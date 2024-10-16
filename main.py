import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists for the data
Intento, Puntaje, Max_Puntaje, Reward_dist, Reward_ctrl = [], [], [], [], []
std_dev_list = []
# Read the CSV file
with open('registro_experimento_final.csv', 'r') as file:
    csv_reader = csv.reader(file)  # No need for delimiter='\t' if it's a CSV

    next(csv_reader)  # Skip the header

    for line in csv_reader:
        # Handle lines correctly, ensuring we join split array entries
        line = ','.join(line).split(',')  # Flatten potential multi-part lines

        try:
            intento = int(line[0])
            puntaje = float(line[3])
            max_puntaje = float(line[4])
            reward_dist = float(line[6])
            reward_ctrl = float(line[7])
            std_dev = float(line[5])
        except (ValueError, IndexError) as e:
            print(f"Error al procesar la línea: {line} - {e}")
            continue

        # Append to corresponding lists
        Intento.append(intento)
        Puntaje.append(puntaje)
        Max_Puntaje.append(max_puntaje)
        Reward_dist.append(reward_dist)
        Reward_ctrl.append(reward_ctrl)
        std_dev_list.append(std_dev)

cumulative_avg = np.cumsum(Puntaje) / np.arange(1, len(Puntaje) + 1)
# Create two figures with two plots each
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8))
fig3, ax3 = plt.subplots(2, 1, figsize=(10, 8))  # Nueva imagen para promedios acumulados y std dev


# Plot 1: Score by Attempt
ax[0].plot(Intento, Puntaje, linestyle='-', label='Puntaje')
ax[0].set_xlabel('Intento')
ax[0].set_ylabel('Puntaje')
ax[0].set_title('Puntaje por Intento')
ax[0].legend()

# Plot 2: Max Score by Attempt
ax[1].plot(Intento, Max_Puntaje, linestyle='-', color='orange', label='Máximo Puntaje')
ax[1].set_xlabel('Intento')
ax[1].set_ylabel('Máximo Puntaje')
ax[1].set_title('Máximo Puntaje por Intento')
ax[1].legend()

# Plot 3: Reward Distance by Attempt
ax2[0].plot(Intento, Reward_dist, linestyle='-', color='green', label='Reward Dist')
ax2[0].set_xlabel('Intento')
ax2[0].set_ylabel('Puntaje Distancia')
ax2[0].set_title('Puntaje Distancia por Intento')
ax2[0].legend()

# Plot 4: Reward Control by Attempt
ax2[1].plot(Intento, Reward_ctrl, linestyle='-', color='blue', label='Reward Ctrl')
ax2[1].set_xlabel('Intento')
ax2[1].set_ylabel('Puntaje Control')
ax2[1].set_title('Puntaje Control por Intento')
ax2[1].legend()

# Gráfico 3: Promedio acumulado de puntajes
ax3[0].plot(Intento, cumulative_avg, linestyle='-', color='blue', label='Promedio Acumulado')
ax3[0].set_xlabel('Intento')
ax3[0].set_ylabel('Promedio Acumulado')
ax3[0].set_title('Promedio Acumulado de Puntajes por Intento')
ax3[0].legend()

# Gráfico 4: Desviación estándar por intento
ax3[1].plot(Intento, std_dev_list, linestyle='-', color='purple', label='Desviación Estándar')
ax3[1].set_xlabel('Intento')
ax3[1].set_ylabel('Desviación Estándar')
ax3[1].set_title('Desviación Estándar por Intento')
ax3[1].legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
