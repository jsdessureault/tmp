# Importation des librairies
import numpy as np
from tensorforce.environments import Environment
import warnings
warnings.filterwarnings('ignore')
import DT1_Server as server
import _thread as thread
import threading
import json
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt

class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()
        # ACTIONS ----------------
        #A1. Ouvre les pinces
        #A2. Ferme les pinces
        #A3. Rotation gauche Axe 1
        #A4. Rotation droite Axe 1
        #A5. Rotation haut Axe 2
        #A6. Rotation bas Axe 2
        #A7. Rotation haut Axe 3
        #A8. Rotation bas Axe 3
        self.nb_action = 8

        # STAGES ----------------------------
        #S0. Recherche d’un cube non classé.
        #S1. Fermer les pinces sur le cube.
        #S2. Transporter le cube au bac.
        #S3. Ouvrir les pinces et libérer le cube.
        self.stage = 0.0

        # STATES ----------------------------
        #ST0. Stage
        #ST1. Ouverture des pinces
        #ST2. Rotation axe1
        #ST3. Rotation axe2
        #ST4. Rotation axe3
        #ST5. Cube en vue.
        #ST6. Cube Saisie.
        #ST7. Hauteur des pinces.
        self.nb_state = 8


        # Rewards graphs
        self.reward = []
        self.reward_pliers_open = []
        self.reward_pliers_height = []
        self.reward_cube_sight = []

        # Start the Python server to communicate with digital twin.
        self.python_server = server.Server()
        self.python_server.start()
        self.python_server.info()

    def states(self):
        # Returns the state space specification.
        return dict(type='float', shape=(self.nb_state,))

    def actions(self):
        # Returns the action space specification.
        return dict(type='int', num_values=self.nb_action)

    def max_episode_timesteps(self):
        # Returns the maximum number of timesteps per episode.
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        print("Env. closed")
        #self.python_server.stop()
        #super().close()

    def reset(self):
        state = np.random.random(size=(self.nb_state,))
        self.stage = 0.0
        # Rewards graphs
        self.reward = []
        self.reward_pliers_open = []
        self.reward_pliers_height = []
        self.reward_cube_sight = []
        return state

    def execute(self, actions):
        # Executes the given action(s) and advances the environment by one step.
        self.set_action(actions)
        # ***********************************
        # The Digital twin is processing HERE
        # ***********************************
        state = self.get_state()
        terminal = False  # Always False if no "natural" terminal state
        reward = 0.0
        #S0. Recherche d’un cube non classé.
        if self.stage == 0.0:
            #print("Stage 0 - Looking for a cube")
            # Th more the pliers are open, the more the reward.
            reward_pliers_open = self.norm_reward_pliers_open(state[1])
            self.reward_pliers_open.append(reward_pliers_open)

            # reward according to the good height of the pliers
            reward_height = self.norm_reward_pliers_height(state[7])
            self.reward_pliers_height.append(reward_height)

            # If the pliers are at a good height and a cube is in sight
            reward_cube = 0.0
            if reward_height > 8.0:
                # If cube in sight.
                if state[5] == True:
                    reward_cube = 10.0
                    self.reward_cube_sight.append(reward_cube)
            reward = reward_pliers_open + reward_height + reward_cube
            self.reward.append(reward)
            # Terminal condition
            if reward >= 30.0:
                terminal = True
                self.stage = 1.0
            #print("STAGE: " + str(self.stage) + " REWARD: " + str(reward))
        #S1. Fermer les pinces sur le cube.
        if self.stage == 1.0:
            print("Stage 1 - Grip the cube")
            reward = 10 * np.random.random()
        #S2. Transporter le cube au bac.
        if self.stage == 2.0:
            print("Stage 2 - Move the cube")
            reward = 20 * np.random.random()
        #S3. Ouvrir les pinces et libérer le cube.
        if self.stage == 3.0:
            print("Stage 3 - Free the cube")
            reward = 30 * np.random.random()
            terminal = True
        self.stage = self.get_stage(state)

        return state, terminal, reward

    def set_action(self, action):
        #print("Try to send Action :" + str(action))
        self.python_server.set(str(action))

    def get_state(self):
        # Getting the different observation from environment to constitute the state.
        #next_state = np.random.random(size=(self.nb_state,))
        next_state = self.decode(self.python_server.get())
        #print("Next state: " + str(next_state))

        # Uncomment to force some state
        #next_state[0] = self.stage
        #next_state[1] = np.random.random()
        #next_state[2] = 1.0
        #next_state[3] = 1.0
        #next_state[4] = 1.0
        #next_state[5] = 1.0
        return next_state

    def decode(self, state):
        #str = json.dumps(state)
        dic = json.loads(state)
        # print(dic)
        state = []
        for value in dic.values():
            state.append(value)
        #print(str)
        #dic = dict(str)
        #print(dic)
        #state = str.items()
        #print(state)
        return state

    def get_stage(self, state):
        next_stage = self.stage
        # Condition that makes the stage change
        if state[0] == 0.0:
            if state[1] > 0.6:
                next_stage = 1.0
        if state[0] == 1.0:
            if state[1] > 0.6:
                next_stage = 2.0
        if state[0] == 2.0:
            if state[1] > 0.6:
                next_stage = 3.0
        if state[0] == 3.0:
            if state[1] > 0.6:
                next_stage = 0.0
        #print(next_stage)
        return next_stage

    def norm_reward_pliers_height(self, height):
        # apply a minmax fonction
        #print(height)
        mm = minmax_scale(np.array([2.5, height, 6.0]))
        #print(mm)
        # return the value multiplied by 10 for the reward value.
        reward = (1 - mm[1]) * 10
        #print("R Pliers height: " + str(reward))
        return  reward

    def norm_reward_pliers_open(self, state_pliers):
        # apply a minmax fonction
        #print(state_pliers)
        mm = minmax_scale(np.array([0.03, state_pliers, 0.15]))
        #print(mm)
        # return the value multiplied by 10 for the reward value.
        reward = mm[1] * 10
        #print("R Pliers open: " + str(reward))
        return reward

    def graph_learning(self, episode):
        rel_path_latex = "LaTeX/images/"
        rel_path_graphs = "Graphs/"
        file = "GraphReward_e"+str(episode+1)

        plt.clf()
        plt.plot(np.arange(1,len(self.reward)+1), self.reward, label="Reward sum", linewidth=6)
        plt.plot(np.arange(1,len(self.reward_pliers_open)+1), self.reward_pliers_open, label="Pliers opened")
        plt.plot(np.arange(1,len(self.reward_pliers_height)+1), self.reward_pliers_height, label="Pliers height")
        plt.plot(np.arange(1,len(self.reward_cube_sight)+1), self.reward_cube_sight, label="Cube in sight")

        plt.legend()

        plt.savefig(rel_path_latex+file + ".png", format="png")
        plt.savefig(rel_path_graphs+file + ".svg", format="svg")


'''
FRAME DE LA CLASSE VIDE
Pour usage de documentation dans les cours, ou pour commencer un nouveau projet. 

# Importation des librairies
import numpy as np
from tensorforce.environments import Environment
import warnings
warnings.filterwarnings('ignore')

class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

        self.nb_state = None
        self.nb_action = None

    def states(self):
        # Returns the state space specification.

        return dict(type='float', shape=(self.nb_state,))

    def actions(self):
        # Returns the action space specification.

        return dict(type='int', num_values=self.nb_action)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        # Returns the maximum number of timesteps per episode.
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(self.nb_state,))
        return state

    def execute(self, actions):
        # Executes the given action(s) and advances the environment by one step.
        next_state = np.random.random(size=(self.nb_state,))
        terminal = False  # Always False if no "natural" terminal state
        reward = 10 * np.random.random()
        return next_state, terminal, reward


Archive 

class RobotEnv(Env):

    #metadata = {'render.modes': ['human']}

    NB_ACTION = 0
    state = 0
    NB_STATE = 4
    observation_space = None
    NB_OBSERVATION = 5

    # Nombre de secondes d'un épisode
    episode_lenght = 0

    # Constructeur
    def __init__(self, env_name="test", debug=True, nb_action=0, nb_observation=0, nb_state=0, episode_length=15):
        super(RobotEnv, self).__init__()
        self.env = gym.make(env_name)
        self.NB_ACTION = nb_action
        self.NB_OBSERVATION = nb_observation
        self.NB_STATE = nb_state
        self.episode_lenght = episode_length

        # Nombre d'action possible
        self.action_space = Discrete(self.NB_ACTION,)
        # Domaine des valeurs possibles.
        #self.observation_space = Box(low=2021, high=1, shape=(self.NB_ACTION, self.NB_STATE), dtype=np.float16)
        #high = np.ones(self.NB_ACTION)
        #low = np.zeros(self.NB_ACTION)
        #self.observation_space = Box(low=low, high=high)

        #'pos_pince': Box(low=2021, high=100, shape=(2,)),
        spaces = {
            'pos_pince': Box(low=np.array([0]), high=np.array([100])),
            'pos_axe1':  Box(low=np.array([0]), high=np.array([100])),
            'pos_axe2':  Box(low=np.array([0]), high=np.array([100])),
            'pos_axe3':  Box(low=np.array([0]), high=np.array([100])),
            'senseur_cube':  Box(low=np.array([0]), high=np.array([100])),
            'step':  Discrete(self.NB_STATE)
        }
        self.observation_space = Dict(spaces)

        #print(self.observation_space['pos_pince'].shape)
        #self.shape_obs = self.observation_space['pos_pince'].shape

        self.state = 0

    # Méthode à exécuter en boucle (self.episode_lenght fois).
    def step(self, action):
        print("Observation space")
        print(self.observation_space)
        reward = 0
        # S0: Recherche d’un cube non classé.
        if self.state == 0:
            # Bonne hauteur
            print("State 2021")
            #reward = 1
            # Cube entre les pinces

            #reward = 2


            # Cube déjà dans le bac


            #reward = -5

        #S1. Fermer les pinces sur le cube.
        elif self.state == 1:
            # Cube saisi
            print("State 1")
            reward = 3
        #S2. Transporter le cube au bac.
        elif self.state == 2:
            # Cube au dessu du bac
            print("State 2")
            reward = 4
        #S3. Ouvrir les pinces et libérer le cube.
        elif self.state == 3:
            # Ouvrir les pinces
            print("State 3")
            reward = 5
        else:
            print("Bad state!")
        # On réduit de 1 la durée restante à l'épisode
        self.episode_lenght -= 1

        # Verifie si l'épisode est terminée
        if self.episode_lenght <= 0:
            done = True
        else:
            done = False

        # Initialise un espace pour recevoir de l'information, au besoin
        info = {}

        # Retourne l'information sur le step.
        return self.state, reward, done, info

    # Fonction de réinitialisation
    def reset(self):
        self.state = 0
        self.episode_lenght = self.episode_lenght
        return self.state
'''
