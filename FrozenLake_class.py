# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
#Import packages
get_ipython().run_line_magic('run', 'imports_gym.ipynb')

# %% [markdown]
# # Q learning algorithm

# %%

class Qvalue:
    '''
    Implements Qvalue learning
    '''    
    def __init__(self, environment, num_episodes, max_steps_per_episode, learning_rate=0.1, discount_rate=0.99,exploration_rate =1, max_exploration_rate = 1, min_exploration_rate   = 0.01, exploration_decay_rate = 0.001):
        self.env                    = gym.make(environment)
        self.num_episodes           = num_episodes
        self.max_steps_per_episode  = max_steps_per_episode 
        self.learning_rate          = learning_rate    
        self.discount_rate          = discount_rate 
        self.exploration_rate       = exploration_rate
        self.max_exploration_rate   = max_exploration_rate 
        self.min_exploration_rate   = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate        
        self.env.reset() #start from the beginning of the episode
        return     
   
    # HELPERS*************************************************************************************************
    def policy(self,state,q_table):
        # Exploration policy -Exploration-exploitation trade-off: decide whether we explore or exploit
        exploration_rate_threshold = random.uniform(0, 1) #pick a rnd value and compare it to the set threshold
        if exploration_rate_threshold > self.exploration_rate:
            action = np.argmax(q_table[state,:]) #exploit and take the max
        else:
            action = self.env.action_space.sample() #explore - sample an action randomly-        
        return action
    
    def updateQ(self,q_table,reward,action,state,new_state):
        # Update Q-table for Q(s,a) - implementation of the Q-value according to the chosen "leaerning rate"
        td_err =  (reward + self.discount_rate * np.max(q_table[new_state, :]))  
        q_table[state, action] = q_table[state, action] * (1 - self.learning_rate) +self.learning_rate *td_err
        return q_table
    
    
    # MAIN *****************************************************************************************************
    def run(self):
        rewards_all_episodes = []  #list container to hold all the rewards across episodes
        q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))#OBS: state_space_size  = env.observation_space.n
            
        for episode in range(self.num_episodes):
            # initialize new episode params 
            state = self.env.reset() #put the agent back on the starting state every time
            done  = False       #the 'done' variable keeps track of whether or not we have finished the episode (i.e. the game)
            rewards_current_episode = 0 #keeps track of the accumulated rewards within the episode
    
            for step in range(self.max_steps_per_episode):             
                #Agent
                action  = self.policy(state,q_table)               
                #Environment             
                new_state, reward, done, info = self.env.step(action) #we store on tuples all the info output by the environment
                #Q-value matrix
                q_table = self.updateQ(q_table,reward,action,state,new_state) 
                #Set new state
                state   = new_state
                #Add new reward     
                rewards_current_episode += reward         
            # Check whether the last step has ended the game (we get this variable from the environment)
            # If the action did end the episode, then we jump out of this loop and move on to the next episode.
            # Otherwise, we transition to the next time-step within the same episode (i.e. the same game) 
                if done == True: 
                    break

        ################### Once the steps are finished ###########################    
        # Exploration rate decay    - this is just a trick to improve performance.
        # We want to explore less and exploit more with time
            self.exploration_rate =  self.min_exploration_rate +(self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate*episode)
                
       # Add current episode reward to total rewards list
            rewards_all_episodes.append(rewards_current_episode)
        
        return  rewards_all_episodes  
    

    

# %% [markdown]
# # Rider

# %%
#env name ####
environment = "FrozenLake-v0"

#define static variables
num_episodes           = 10000 #number of games to play
max_steps_per_episode  = 100   #how many steps on each game
learning_rate          = 0.1   #how much we want to update the Q-value at each stage.
discount_rate          = 0.99  #how much weight we give to future rewards

#### experimental values ####
exploration_rate       = 1
max_exploration_rate   = 1
min_exploration_rate   = 0.01
exploration_decay_rate = 0.001  #this is experimental.. change and see what happens.

qval                 = Qvalue(environment, num_episodes, max_steps_per_episode, learning_rate, discount_rate,exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate)
rewards_all_episodes = qval.run()

#print(rewards_all_episodes)


# %%
# Calculate and print the average reward per thousand episodes

rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("********Average reward per thousand episodes (in percentage)********\n") ## we want the reward to improve over time
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000 
    

# Print updated Q-table
#print("\n\n********Q-table********\n")
#print(q_table)  ### TODO


# %%
# Vusualization


# %%
# Watch our agent play Frozen Lake by playing the best action 
# from each state according to the Q-table

for episode in range(3):
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):        
        clear_output(wait=True) #function from Jupyter that clears the current cell in the notebook
        env.render()     # renders the current state of the game to the display - to see where in the board we are.
        time.sleep(0.3) #to give time to see the movement on the screen before moving to the next step
        
        action = np.argmax(q_table[state,:])     # take the action with the highest Q value   
        new_state, reward, done, info = env.step(action) #execute action and take response from Environment
        
        if done: #if the game ended, it's either because either the Agent reached the goal or it fell throu a hole
            clear_output(wait=True)
            env.render() # renders the current state of the game to the display - to see where in the board we are.
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
            clear_output(wait=True)
            break #if last action ended the game, stop the code
            
        state = new_state  #if our last Action didn't end the game, transition to the new state.
        
env.close() # at the very end, close the environment.

