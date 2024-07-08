import matplotlib.pyplot as plt

import numpy as np

def plot_job_scheduling(job_schedule, n, m):
    machine_schedule = {}
    l = []
    l1 = []
    temp = []

    for i in range(len(job_schedule)):
        l.append(job_schedule[i])

    for i in range(m):
        for j in range(n):
            temp.append(l[j][i])
        l1.append(temp)
        temp = []

    for i in range(len(l1)):
        machine_schedule[i] = l1[i]

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)

    # Set y-axis label and limits
    ax.set_ylabel('Machine', fontweight='bold', loc='top', color='magenta', fontsize=12)
    ax.set_ylim(-0.5, len(machine_schedule))
    ax.tick_params(axis='y', labelcolor='magenta', labelsize=12)

    # Set x-axis label and limits
    ax.set_xlabel('Time', fontweight='bold', loc='right', color='red', fontsize=12)
    ax.tick_params(axis='x', labelcolor='red', labelsize=12)

    ax.grid(True)

    # Create a color list for the jobs
    colors = ['orange', 'deepskyblue', 'indianred', 'limegreen', 'slateblue', 'gold', 'violet', 'grey', 'red', 'magenta']

    job_id = 0

    for i, (id, schedule) in enumerate(machine_schedule.items()):
        
        for machine_id, (start_time, end_time) in enumerate(schedule):
            job_id+=1
            duration = end_time - start_time

            # Calculate the x-position and y-position for the bar
            x_pos = start_time
            y_pos = i - 0.3

            # Choose color based on machine_id
            color_idx = machine_id % len(colors)

            # Plot the bar
            ax.broken_barh([(x_pos, duration)], (y_pos, 0.6), facecolor=colors[color_idx], linewidth=1, edgecolor='black')

            # Add job label
            label_x_pos = x_pos + (duration / 2)
            label_y_pos = i + 0.03
            ax.text(label_x_pos, label_y_pos, str(job_id), fontsize=10, ha='center')
        job_id = 0

    plt.title('Job Scheduling', size=14, color='blue')
    plt.savefig('gannt.png')


def plot_fitness(iterations, fitness_values, n, m, N, makespan):
    plt.figure(2)
    plt.plot(iterations, fitness_values, label="Fitness_n{}_m{}_N{}".format(n, m, N))
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness Values')
    plt.title('Cmax = {}'.format(makespan))
    plt.grid(True)
    plt.legend()
    plt.savefig('best_fitness_iter.png')