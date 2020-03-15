import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#solving blackjack using on-policy monte carlo estimates 
#GOAL: find optimal policy for this environment 
#credit: Jeffrey Zhang

def deal_card():  
    #(J, Q, K) are noted as 10 
    #choose between 
    JQK_list = [10, 10, 10]
    cards_to_choose = list(range(1, 11)) + JQK_list
    return np.random.choice(cards_to_choose) 


class Blackjack():
    # things we need to keep track of 
    def __init__(self): 
        self.player_state_values = {}  
        self.player_states = []  
        #keep track of win and draw 
        self.win = 0
        self.draw = 0  

    def policy(self, curr_val, usable_ace, done, stand_val):  
        if (curr_val) > 21: 
            curr_val -= 10 
            usable_ace = False
            return curr_val, usable_ace, True

        if curr_val >= stand_val: 
            return curr_val, usable_ace, True
        else: 
            card = deal_card() 
            if (card == 1): 
                if (curr_val <= 10): 
                    curr_val += 11 
                    return curr_val, True, False  
                else:
                    curr_val += 1
                    return curr_val, usable_ace, False 
            else: 
                curr_val += card  
                return curr_val, usable_ace, False  

        return curr_val, usable_ace, done 
    
        
    #reward _function
    def reward(self, player_val, dealer_val, done=True): 
        #check win or draw 
        #reward will be given to the state that wins the game
        if done: 
            last_state = self.player_states[-1] 
            if player_val > 21:  
                if dealer_val > 21:    
                    self.draw += 1 
                else:  
                    #player loss 
                    self.player_state_values[last_state] -= 1 
            else:  
                if dealer_val > 21: 
                    self.win += 1 
                    self.player_state_values[last_state] += 1  
                else: 
                    if player_val > dealer_val: 
                        self.win += 1 
                        self.player_state_values[last_state] += 1
                    elif player_val < dealer_val: 
                        self.player_state_values[last_state] -= 1 
                    else:
                        self.draw += 1

 
    def mc_blackjack(self, n_iterations):  
        rewards_list = []
        
        for i in range(n_iterations):
            if (i % 1000) == 0: 
                print("round:", i)
                
            player_val = 0
            dealer_val = 0  
            show_card = 0 
            
            #generating an episode 
        
            #1) deal 2 cards, show 1
            dealer_val += deal_card() 
            show_card = dealer_val 
            dealer_val += deal_card()  

            #2) player plays
            usable_ace = False 
            done = False 
            # players turn  
            while True:  
                (player_val, usable_ace, done) = self.policy(player_val, usable_ace, done, 20) 

                if done: 
                    break 
                #need to record states for which the players cards are between 12-21
                if (player_val >= 12) and (player_val <= 21): 
                    self.player_states.append((player_val, show_card, usable_ace)) 
               
            #dealers turn
            done = False 
            usable_ace = False
            while not done:  
                (dealer_val, usable_ace, done) = self.policy(dealer_val, usable_ace, done, 17) 
 
            #set rewards for all non-terminal states to 0 
            for s in self.player_states:
                self.player_state_values[s] = 0 if self.player_state_values.get(s) is None else self.player_state_values.get(s)

            #assign reward to winner 
            self.reward(player_val, dealer_val)         
            
            #store rewards 
            if (i % 100) == 0: 
                last_state = self.player_states[-1]
                reward_episode = self.player_state_values[last_state]   
                print("reward accumulated: {}".format(reward_episode))
                rewards_list.append(reward_episode)  
        
        return rewards_list

if __name__ == "__main__":
    env = Blackjack()
    n_iterations = 10000  
    rewards_list = env.mc_blackjack(n_iterations) 
    
    # data
    print("The player won {} umber of times and drew {} number of times".format(env.win, env.draw)) 
    probs_winning = float(env.win) / float(n_iterations)
    print("Probability of winning: {}%".format(probs_winning*100)) 
    probs_draw = float(env.draw) / float(n_iterations)
    print("Probability of draw: {}%".format(probs_draw*100)) 
    
    #plotting 
    fig = plt.figure(figsize=[15, 6])
    ax1 = fig.add_subplot(121, projection='3d')

    x1 = [k[1] for k in env.player_state_values.keys()]
    y1 = [k[0] for k in env.player_state_values.keys()] 
    z1 = [v for v in env.player_state_values.values()]
    ax1.scatter(x1, y1, z1)

    ax1.set_title("Game of Blackjack")
    ax1.set_xlabel("dealer showing")
    ax1.set_ylabel("player sum")
    ax1.set_zlabel("reward")

    plt.show()
            
            
        
               
    