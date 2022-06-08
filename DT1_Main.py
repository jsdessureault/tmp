from DT1.DT1_Env import CustomEnvironment
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from matplotlib import pyplot as plt
import numpy as np

# Doc: https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html
# Code avec exemples: https://github.com/tensorforce/tensorforce

def DRL():
    print("Deep reinforcement learning for Digital Twin")

    # EPISODES-----------------------
    EPISODE_LENGHT = 100    # 100
    NB_EPISODES = 12       # 10

    # Environement creation
    environment = Environment.create(
        environment=CustomEnvironment,
        max_episode_timesteps=EPISODE_LENGHT
        )

    # Agent creation
    agent = Agent.create(
        agent='tensorforce',
        environment=environment,
        update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient',
        reward_estimation=dict(horizon=20)
    )

    agent.reset()
    graph_reward_overall = []
    for i in range(NB_EPISODES):
        print("Executing episode " + str(i) + "...")
        states = environment.reset()
        sum_rewards = 0.0
        num_updates = 0.0
        terminal = False
        while not terminal:
            #for j in range(3):
            actions = agent.act(states=states, deterministic=False)     # Deterministic False:  Allow exploration
            states, terminal, reward = environment.execute(actions=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward)     # parallele:  nb.core?
            sum_rewards += reward
        graph_reward_overall.append(sum_rewards)
        print('Episode {}: Sum of the rewards={}'.format(i, sum_rewards))
        environment.graph_learning(i)
    graph_learning(graph_reward_overall)

    environment.close()
    agent.close()
    print("Process ended properly.")

def graph_learning(reward):
    rel_path_latex = "LaTeX/images/"
    rel_path_graphs = "Graphs/"
    file = "GraphReward_total"

    plt.clf()
    plt.plot(np.arange(1,len(reward)+1), reward)
    plt.savefig(rel_path_latex+file + ".png", format="png")
    plt.savefig(rel_path_graphs+file + ".svg", format="svg")


# Todo: X - Copier les prefab de UPython2 dans le jumeau numérique
# Todo: X - Démarrer le jumeaux avec script
# Todo: X - Assurer la communication en alternance
# Todo: X - Assurer la communication en alternance dans le contexte de DT1.
# Todo: X - Décoder l'info côté jumeaux.
# Todo: X - Receuillir l'action du serveur et l'envoyer au jumeaux.
# Todo: X - Décoder l'info côté serveur.
# Todo: X - Recueillir les données de l'environnement du jumeaux et l'envoyer au serveur.
# Todo: X - Recevoir l'action côté client
# Todo: X - Améliorer la fluidité de l'affichage a la console.
# Todo: X - Executer l'action sur le jumeau.
# Todo: X - Avoir les bons states de retour.
# Todo: X - S'assurer que le graphique fonctionne
# Todo: Programmation des stages.

DRL()

'''
    FRAME DE LA CLASSE VIDE
    Pour usage de documentation dans les cours, ou pour commencer un nouveau projet. 

    print("Deep reinforcement learning for Digital Twin")

    EPISODE_LENGHT = 100
    NB_EPISODES = 10

    # Environement creation
    environment = Environment.create(
        environment=CustomEnvironment,
        max_episode_timesteps=EPISODE_LENGHT
        )

    environment.set_nb_actions(NB_ACTION)
    environment.set_nb_states(NB_STATE)

    # Agent creation
    agent = Agent.create(
        agent='tensorforce',
        environment=environment,
        update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient',
        reward_estimation=dict(horizon=20)
    )

    for i in range(NB_EPISODES):
        print("Episode:" + str(i))
        states = environment.reset()
        terminal = False
        graph_reward = []
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            graph_reward.append(reward)
        graph_learning(i, graph_reward)

    agent.close()
    environment.close()

def graph_learning(episode, reward):
    rel_path_latex = "LaTeX/images/"
    rel_path_graphs = "Graphs/"
    file = "GraphReward_e"+str(episode+1)

    plt.clf()
    plt.plot(np.arange(1,len(reward)+1), reward)
    plt.savefig(rel_path_latex+file + ".png", format="png")
    plt.savefig(rel_path_graphs+file + ".svg", format="svg")
'''