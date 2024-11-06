import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists for the data
Intento, Puntaje, Max_Puntaje, Reward_dist, Reward_ctrl = [], [], [], [], []
Reward_near_object, Reward_contact, Episode_length, TD_error, Actor_loss, Critic_loss, Policy_gradients = [], [], [], [], [], [], []
std_dev_list = []
Duracion = []
# Read the CSV file
with open('registro_experimento_final.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header

    for line in csv_reader:
        # Handle lines correctly, ensuring we join split array entries
        line = ','.join(line).split(',')  # Flatten potential multi-part lines

        try:
            intento = int(line[0])
            puntaje = float(line[3])
            max_puntaje = float(line[4])
            reward_dist = float(line[5])
            reward_ctrl = float(line[6])
            reward_near_object = float(line[7])
            reward_contact = float(line[8])
            episode_length = int(line[9])
            td_error = float(line[10])
            actor_loss = float(line[11])
            critic_loss = float(line[12])
            policy_gradients = float(line[13])
            duracion = float(line[2])  # Agregar duración del episodio
        except (ValueError, IndexError) as e:
            print(f"Error al procesar la línea: {line} - {e}")
            continue

        # Append to corresponding lists
        Intento.append(intento)
        Puntaje.append(puntaje)
        Max_Puntaje.append(max_puntaje)
        Reward_dist.append(reward_dist)
        Reward_ctrl.append(reward_ctrl)
        Reward_near_object.append(reward_near_object)
        Reward_contact.append(reward_contact)
        Episode_length.append(episode_length)
        TD_error.append(td_error)
        Actor_loss.append(actor_loss)
        Critic_loss.append(critic_loss)
        Policy_gradients.append(policy_gradients)
        Duracion.append(duracion)

cumulative_avg = np.cumsum(Puntaje) / np.arange(1, len(Puntaje) + 1)

# Initialize lists for the data from best_scores_final.csv
Best_Score_Attempts, Best_Scores = [], []

# Read the best_scores_final.csv file
with open('best_scores_final.csv', 'r') as best_file:
    best_csv_reader = csv.reader(best_file)
    next(best_csv_reader)  # Skip the header

    for best_line in best_csv_reader:
        try:
            best_intento = int(best_line[0])  # Attempt number
            best_score = float(best_line[1])  # Score
        except (ValueError, IndexError) as e:
            print(f"Error al procesar la línea de best_scores_final: {best_line} - {e}")
            continue

        # Append to corresponding lists
        Best_Score_Attempts.append(best_intento)
        Best_Scores.append(best_score)


# Calculate the average every 100 episodes
avg_reward_dist = [np.mean(Reward_dist[i:i+100]) for i in range(0, len(Reward_dist), 100)]
avg_reward_ctrl = [np.mean(Reward_ctrl[i:i+100]) for i in range(0, len(Reward_ctrl), 100)]
avg_intentos = [np.mean(Intento[i:i+100]) for i in range(0, len(Intento), 100)]
avg_reward_near_object = [np.mean(Reward_near_object[i:i+100]) for i in range(0, len(Reward_near_object), 100)]
avg_reward_contact = [np.mean(Reward_contact[i:i+100]) for i in range(0, len(Reward_contact), 100)]
avg_td_error = [np.mean(TD_error[i:i+100]) for i in range(0, len(TD_error), 100)]
avg_actor_loss = [np.mean(Actor_loss[i:i+100]) for i in range(0, len(Actor_loss), 100)]
avg_critic_loss = [np.mean(Critic_loss[i:i+100]) for i in range(0, len(Critic_loss), 100)]
avg_policy_gradients = [np.mean(Policy_gradients[i:i+100]) for i in range(0, len(Policy_gradients), 100)]

avg_best_score_a = [np.mean(Best_Score_Attempts[i:i+5]) for i in range(0, len(Best_Score_Attempts), 5)]
avg_best_score = [np.mean(Best_Scores[i:i+5]) for i in range(0, len(Best_Scores), 5)]

avg_normal_score_a = [np.mean(Intento[i:i+100]) for i in range(0, len(Intento), 100)]
avg_normal_score = [np.mean(Puntaje[i:i+100]) for i in range(0, len(Puntaje), 100)]

# Create figures for the plots
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(2, 2, figsize=(14, 12))  # Updated to 2 rows and 2 columns
fig3, ax3 = plt.subplots(2, 1, figsize=(10, 8))
fig4, ax4 = plt.subplots(2, 1, figsize=(10, 8))
fig5, ax5 = plt.subplots(2, 1, figsize=(10, 8))

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
ax2[0, 0].plot(avg_intentos, avg_reward_dist, linestyle='-', color='green', label='Reward Dist')
ax2[0, 0].set_xlabel('Intento')
ax2[0, 0].set_ylabel('Puntaje Distancia')
ax2[0, 0].set_title('Puntaje Promedio Distancia por cada 100 intentos')
ax2[0, 0].legend()

# Plot 4: Reward Control by Attempt
ax2[0, 1].plot(avg_intentos, avg_reward_ctrl, linestyle='-', color='blue', label='Reward Ctrl')
ax2[0, 1].set_xlabel('Intento')
ax2[0, 1].set_ylabel('Puntaje Control')
ax2[0, 1].set_title('Puntaje Promedio Control por cada 100 intentos')
ax2[0, 1].legend()

# Plot 5: Reward Near Object by Attempt
ax2[1, 0].plot(avg_intentos, avg_reward_near_object, linestyle='-', color='red', label='Reward Near Object')
ax2[1, 0].set_xlabel('Intento')
ax2[1, 0].set_ylabel('Recompensa Cerca del Objeto')
ax2[1, 0].set_title('Recompensa Promedio por Acercarse al Objeto por cada 100 intentos')
ax2[1, 0].legend()

# Plot 6: Reward Contact by Attempt
ax2[1, 1].plot(avg_intentos, avg_reward_contact, linestyle='-', color='purple', label='Reward Contact')
ax2[1, 1].set_xlabel('Intento')
ax2[1, 1].set_ylabel('Recompensa Contacto')
ax2[1, 1].set_title('Recompensa Promedio por Contacto Exitoso por cada 100 intentos')
ax2[1, 1].legend()

# Gráfico 3: Promedio acumulado de puntajes
ax3[0].plot(Intento, cumulative_avg, linestyle='-', color='blue', label='Promedio Acumulado')
ax3[0].set_xlabel('Intento')
ax3[0].set_ylabel('Promedio Acumulado')
ax3[0].set_title('Promedio Acumulado de Puntajes por Intento')
ax3[0].legend()

# Plot 8: Best Scores from best_scores_final.csv
ax3[1].plot(Best_Score_Attempts, Best_Scores, linestyle='-', color='purple', label='Best Scores from File')
ax3[1].set_xlabel('Intento')
ax3[1].set_ylabel('Best Scores')
ax3[1].set_title('Best Scores by Attempt (from best_scores_final.csv)')
ax3[1].legend()

# Gráfico 4: Promedio acumulado de puntajes
ax4[0].plot(avg_best_score_a, avg_best_score, linestyle='-', color='blue', label='Promedio Mejores Puntajes')
ax4[0].set_xlabel('Intentos x 3')
ax4[0].set_ylabel('Mejor Puntaje Promedio x3')
ax4[0].set_title('Promedio Mejores Puntajes por c/ 3 intentos - Best')
ax4[0].legend()

# Plot 10: Best Scores from best_scores_final.csv
ax4[1].plot(avg_normal_score_a, avg_normal_score, linestyle='-', color='purple', label='Promedio x cada 100 episodios')
ax4[1].set_xlabel('Intento')
ax4[1].set_ylabel('Puntaje Promedio')
ax4[1].set_title('Puntaje Promedio por cada 100 episodios - Normal')
ax4[1].legend()

# Plot 11: Tiempo por cada episodio
ax5[0].plot(Intento, Duracion, linestyle='-', color='black', label='Duración del Episodio')
ax5[0].set_xlabel('Intento')
ax5[0].set_ylabel('Duración (s)')
ax5[0].set_title('Duración del Episodio por Intento')
ax5[0].legend()

# Plot 12: Best Scores
ax5[1].plot(Best_Score_Attempts, Best_Scores, linestyle='-', color='cyan', label='Best Scores')
ax5[1].set_xlabel('Intento')
ax5[1].set_ylabel('Best Scores')
ax5[1].set_title('Todos los Puntajes de Best Scores')
ax5[1].legend()


# Adjust layout and show the plots
plt.tight_layout()
plt.show()
