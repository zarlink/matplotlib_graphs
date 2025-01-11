import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists for the data
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize lists for the data
Puntaje, Max_Puntaje, Reward_dist, Reward_ctrl = [], [], [], []
Reward_near_object, Reward_contact, Episode_length, TD_error_1, TD_error_2, Actor_loss, Critic_loss_1, Critic_loss_2, Policy_gradients = [], [], [], [], [], [], [], [], []
Duracion , Distance_moved , Distance_left = [],[],[]


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
            td_error_1 = float(line[10])
            td_error_2 = float(line[11])
            actor_loss = float(line[12])
            critic_loss_1 = float(line[13])
            critic_loss_2 = float(line[14])
            policy_gradients = float(line[15])
            duracion = float(line[2])  # Agregar duración del episodio
            distance_moved = float(line[16])
            distance_left = float(line[17])
        except (ValueError, IndexError) as e:
            print(f"Error al procesar la línea: {line} - {e}")
            continue

        # Append to corresponding lists
        #Intento.append(intento)
        Puntaje.append(puntaje)
        Max_Puntaje.append(max_puntaje)
        Reward_dist.append(reward_dist)
        Reward_ctrl.append(reward_ctrl)
        Reward_near_object.append(reward_near_object)
        Reward_contact.append(reward_contact)
        Episode_length.append(episode_length)
        TD_error_1.append(td_error_1)
        TD_error_2.append(td_error_2)
        Actor_loss.append(actor_loss)
        Critic_loss_1.append(critic_loss_1)
        Critic_loss_2.append(critic_loss_2)
        Policy_gradients.append(policy_gradients)
        Duracion.append(duracion)
        Distance_moved.append(distance_moved)
        Distance_left.append(distance_left)

Intento = list(range(1, len(Puntaje) + 1))
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
        Best_Score_Attempts.append(len(Best_Score_Attempts) + 1)
        Best_Scores.append(best_score)

# Calculate the average every 100 episodes
avg_reward_dist = [np.mean(Reward_dist[i:i+100]) for i in range(0, len(Reward_dist), 100)]
avg_reward_ctrl = [np.mean(Reward_ctrl[i:i+100]) for i in range(0, len(Reward_ctrl), 100)]
avg_intentos = [np.mean(Intento[i:i+100]) for i in range(0, len(Intento), 100)]
avg_reward_near_object = [np.mean(Reward_near_object[i:i+100]) for i in range(0, len(Reward_near_object), 100)]
avg_reward_contact = [np.mean(Reward_contact[i:i+100]) for i in range(0, len(Reward_contact), 100)]
avg_td_error_1 = [np.mean(TD_error_1[i:i+100]) for i in range(0, len(TD_error_1), 100)]
avg_td_error_2 = [np.mean(TD_error_2[i:i+100]) for i in range(0, len(TD_error_2), 100)]
avg_actor_loss = [np.mean(Actor_loss[i:i+100]) for i in range(0, len(Actor_loss), 100)]
avg_critic_loss_1 = [np.mean(Critic_loss_1[i:i+100]) for i in range(0, len(Critic_loss_1), 100)]
avg_critic_loss_2 = [np.mean(Critic_loss_2[i:i+100]) for i in range(0, len(Critic_loss_2), 100)]
avg_policy_gradients = [np.mean(Policy_gradients[i:i+100]) for i in range(0, len(Policy_gradients), 100)]
avg_best_score_a = [np.mean(Best_Score_Attempts[i:i+5]) for i in range(0, len(Best_Score_Attempts), 5)]
avg_best_score = [np.mean(Best_Scores[i:i+5]) for i in range(0, len(Best_Scores), 5)]
avg_normal_score_a = [np.mean(Intento[i:i+100]) for i in range(0, len(Intento), 100)]
avg_normal_score = [np.mean(Puntaje[i:i+100]) for i in range(0, len(Puntaje), 100)]


# Create figures for the plots, organizing them into groups of 4 per image
figures = []

# Plot definitions grouped by 4 per figure
for i in range(0, 11, 4):
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))

    if i == 0:
        # Plot 1: Score by Attempt
        ax[0, 0].plot(Intento, Puntaje, linestyle='-', label='Score')
        ax[0, 0].set_xlabel('Attempt')
        ax[0, 0].set_ylabel('Score')
        ax[0, 0].set_title('Score per attempt')
        ax[0, 0].legend()

        ax[0, 1].plot(Intento, Reward_contact, linestyle='-', color='red', label='Distance Arm - Obj ')
        ax[0, 1].set_xlabel('Attempt')
        ax[0, 1].set_ylabel('Distance (m)')
        ax[0, 1].set_title('Distance arm to objective')
        ax[0, 1].legend()

        # Plot 3: Reward Distance by Attempt
        ax[1, 0].plot(Intento, Distance_moved, linestyle='-', color='green', label='Distance Moved by the object')
        ax[1, 0].set_xlabel('Attempt')
        ax[1, 0].set_ylabel('Distance Displaced')
        ax[1, 0].set_title('Distance displaced by the object')
        ax[1, 0].legend()

        # Plot 4: Reward Control by Attempt
        ax[1, 1].plot(Intento, Distance_left, linestyle='-', color='blue', label='Distance Left To Goal')
        ax[1, 1].set_xlabel('Attempt')
        ax[1, 1].set_ylabel('Distance Left')
        ax[1, 1].set_title('Distance Left to Goal')
        ax[1, 1].legend()

    elif i == 4:
        # Plot 6: Promedio acumulado de puntajes
        ax[0, 0].plot(avg_intentos, avg_reward_ctrl, linestyle='-', color='blue', label='Reward Ctrl')
        ax[0, 0].set_xlabel('Attempt')
        ax[0, 0].set_ylabel('Control Score')
        ax[0, 0].set_title('Control Score per 100 attempts')
        ax[0, 0].legend()

        # Plot 7: Average Critic Losses by Attempt
        ax[0, 1].plot(avg_intentos, avg_actor_loss, linestyle='-', color='brown', label='Actor Loss')
        ax[0, 1].set_xlabel('Attempt')
        ax[0, 1].set_ylabel('Actor Loss')
        ax[0, 1].set_title('Actor Loss per 100 attempts')
        ax[0, 1].legend()

        # Plot 8: Critic Loss 1 by Attempt
        ax[1, 0].plot(avg_intentos, avg_critic_loss_1, linestyle='-', color='purple', label='Critic Loss 1')
        ax[1, 0].set_xlabel('Attempt')
        ax[1, 0].set_ylabel('Critic Loss 1')
        ax[1, 0].set_title('Average Critic Loss 1 per 100 attempts')
        ax[1, 0].legend()

        # Plot 9: Promedio Mejores Puntajes por c/3 intentos
        ax[1, 1].plot(avg_intentos, avg_critic_loss_2, linestyle='-', color='purple', label='Critic Loss 2')
        ax[1, 1].set_xlabel('Attempt')
        ax[1, 1].set_ylabel('Crític Loss 2')
        ax[1, 1].set_title('Average Crític Loss 2 per 100 attempts')
        ax[1, 1].legend()

    elif i == 8:
        # Plot 13: TD Error 1 by Attempt
        ax[0, 0].plot(avg_intentos, avg_td_error_1, linestyle='-', color='pink', label='TD Error 1')
        ax[0, 0].set_xlabel('Attempt')
        ax[0, 0].set_ylabel('TD Error 1')
        ax[0, 0].set_title('Average TD Error 1  per 100 attempts')
        ax[0, 0].legend()

        # Plot 14: TD Error 2 by Attempt
        ax[0, 1].plot(avg_intentos, avg_td_error_2, linestyle='-', color='purple', label='TD Error 2')
        ax[0, 1].set_xlabel('Attempt')
        ax[0, 1].set_ylabel('TD Error 2')
        ax[0, 1].set_title('Average TD Error 2 per 100 attempts')
        ax[0, 1].legend()

        # Plot 15: Critic Loss by Attempt
        ax[1, 0].plot(avg_intentos, avg_actor_loss, linestyle='-', color='brown', label='Actor Loss')
        ax[1, 0].set_xlabel('Attempt')
        ax[1, 0].set_ylabel('Actor Loss')
        ax[1, 0].set_title('Average Actor Loss per 100 attempts')
        ax[1, 0].legend()

        # Policy Gradients by Attempt
        ax[1, 1].plot(avg_intentos, avg_policy_gradients, linestyle='-', color='orange', label='Policy Gradients')
        ax[1, 1].set_xlabel('Attempt')
        ax[1, 1].set_ylabel('Policy Gradient')
        ax[1, 1].set_title('Average Policy Gradient per 100 attempts')
        ax[1, 1].legend()

    # Maximizar las ventanas antes de mostrar
    plt.tight_layout()
    figures.append(fig)

# Display all figures
plt.show()
