//
//  tutorial.cpp
//  RLTutorial
//
//  Created by Julio Godoy on 11/25/18.
//  Copyright © 2018 Julio Godoy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string.h>

using namespace std;

int height_grid, width_grid, action_taken, action_taken2, current_episode;
int maxA[100][100], blocked[100][100];
float maxQ[100][100], cum_reward, Qvalues[100][100][4], reward[100][100], finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos, y_pos, prev_x_pos, prev_y_pos, blockedx, blockedy, i, j, k;
ofstream reward_output;

//////////////
// Setting value for learning parameters
int action_sel = 2;         // 1 is greedy, 2 is e-greedy
int environment = 2;        // 1 is small grid, 2 is Cliff walking
int algorithm = 2;          // 1 is Q-learning, 2 is Sarsa
int stochastic_actions = 0; // 0 is deterministic actions, 1 for stochastic actions
int num_episodes = 3000;    // total learning episodes
float learn_rate = 0.1;     // how much the agent weights each new sample
float disc_factor = 0.99;   // how much the agent weights future rewards
float exp_rate = 0.05;      // how much the agent explores
///////////////

void Initialize_environment()
{
    if (environment == 1)
    {

        height_grid = 3;
        width_grid = 4;
        goalx = 3;
        goaly = 2;
        init_x_pos = 0;
        init_y_pos = 0;
    }

    if (environment == 2)
    {

        height_grid = 4;
        width_grid = 12;
        goalx = 11;
        goaly = 0;
        init_x_pos = 0;
        init_y_pos = 0;
    }

    for (i = 0; i < width_grid; i++)
    {
        for (j = 0; j < height_grid; j++)
        {

            if (environment == 1)
            {
                reward[i][j] = -0.04; //-1 if environment 2
                blocked[i][j] = 0;
            }

            if (environment == 2)
            {
                reward[i][j] = -1;
                blocked[i][j] = 0;
            }

            for (k = 0; k < 4; k++)
            {
                Qvalues[i][j][k] = 0; // Initialize Q-values to 0 instead of random
            }
        }
    }

    if (environment == 1)
    {
        reward[goalx][goaly] = 100;
        reward[goalx][(goaly - 1)] = -100;
        blocked[1][1] = 1;
    }

    if (environment == 2)
    {
        reward[goalx][goaly] = 1;

        for (int h = 1; h < goalx; h++)
        {
            reward[h][0] = -100;
        }
    }
}

int action_selection()
{ // Based on the action selection method chosen, it selects an action to execute next

    if (action_sel == 1) // Greedy, always selects the action with the largest Q value
    {
        float max_q = Qvalues[x_pos][y_pos][0];
        int best_action = 0;
        for (int a = 1; a < 4; a++)
        {
            if (Qvalues[x_pos][y_pos][a] > max_q)
            {
                max_q = Qvalues[x_pos][y_pos][a];
                best_action = a;
            }
        }
        return best_action;
    }

    if (action_sel == 2) // epsilon-greedy, selects the action with the largest Q value with prob (1-exp_rate) and a random action with prob (exp_rate)
    {
        if ((float)rand() / RAND_MAX < exp_rate)
        {
            return rand() % 4; // Explore: random action
        }
        else
        {
            // Exploit: greedy action
            float max_q = Qvalues[x_pos][y_pos][0];
            int best_action = 0;
            for (int a = 1; a < 4; a++)
            {
                if (Qvalues[x_pos][y_pos][a] > max_q)
                {
                    max_q = Qvalues[x_pos][y_pos][a];
                    best_action = a;
                }
            }
            return best_action;
        }
    }
    return 0;
}

void move(int action)
{
    prev_x_pos = x_pos; // Backup of the current position, which will become past position after this method
    prev_y_pos = y_pos;

    // Stochastic transition model (not known by the agent)
    // Assuming a .8 prob that the action will perform as intended, 0.1 prob. of moving instead to the right, 0.1 prob of moving instead to the left

    if (stochastic_actions)
    {
        float prob = (float)rand() / RAND_MAX;
        if (prob > 0.8)
        { // 20% chance of not executing intended action
            if (prob <= 0.9)
            { // 10% chance of moving right of intended direction
                action = (action + 1) % 4;
            }
            else
            {                              // 10% chance of moving left of intended direction
                action = (action + 3) % 4; // equivalent to (action - 1) % 4 but handles negative
            }
        }
        // 80% chance: action remains unchanged
    }

    // After determining the real outcome of the chosen action, move the agent

    if (action == 0) // Up
    {

        if ((y_pos < (height_grid - 1)) && (blocked[x_pos][y_pos + 1] == 0)) // If there is no wall or obstacle Up from the agent
        {
            y_pos = y_pos + 1; // move up
        }
    }

    if (action == 1) // Right
    {

        if ((x_pos < (width_grid - 1)) && (blocked[x_pos + 1][y_pos] == 0)) // If there is no wall or obstacle Right from the agent
        {
            x_pos = x_pos + 1; // Move right
        }
    }

    if (action == 2) // Down
    {

        if ((y_pos > 0) && (blocked[x_pos][y_pos - 1] == 0)) // If there is no wall or obstacle Down from the agent
        {
            y_pos = y_pos - 1; // Move Down
        }
    }

    if (action == 3) // Left
    {

        if ((x_pos > 0) && (blocked[x_pos - 1][y_pos] == 0)) // If there is no wall or obstacle Left from the agent
        {
            x_pos = x_pos - 1; // Move Left
        }
    }
}

void update_q_prev_state() // Updates the Q value of the previous state
{

    // Determine the max_a(Qvalue[x_pos][y_pos])
    float max_q_next = Qvalues[x_pos][y_pos][0];
    for (int a = 1; a < 4; a++)
    {
        if (Qvalues[x_pos][y_pos][a] > max_q_next)
        {
            max_q_next = Qvalues[x_pos][y_pos][a];
        }
    }

    // Update the Q value of the previous state and action if the agent has not reached a terminal state
    if (!(((x_pos == goalx) && (y_pos == goaly)) || ((environment == 1) && (x_pos == goalx) && (y_pos == (goaly - 1))) || ((environment == 2) && (x_pos > 0) && (x_pos < goalx) && (y_pos == 0))))
    {
        // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max_a'(Q(s',a')) - Q(s,a)]
        Qvalues[prev_x_pos][prev_y_pos][action_taken] = Qvalues[prev_x_pos][prev_y_pos][action_taken] +
                                                        learn_rate * (reward[x_pos][y_pos] + disc_factor * max_q_next - Qvalues[prev_x_pos][prev_y_pos][action_taken]);
    }
    else // Update the Q value of the previous state and action if the agent has reached a terminal state
    {
        // Terminal state: Q(s,a) = Q(s,a) + α[r - Q(s,a)]
        Qvalues[prev_x_pos][prev_y_pos][action_taken] = Qvalues[prev_x_pos][prev_y_pos][action_taken] +
                                                        learn_rate * (reward[x_pos][y_pos] - Qvalues[prev_x_pos][prev_y_pos][action_taken]);
    }
}

void update_q_prev_state_sarsa()
{
    // Update the Q value of the previous state and action if the agent has not reached a terminal state
    if (!(((x_pos == goalx) && (y_pos == goaly)) || ((environment == 1) && (x_pos == goalx) && (y_pos == (goaly - 1))) || ((environment == 2) && (x_pos > 0) && (x_pos < goalx) && (y_pos == 0))))
    {
        // SARSA update: Q(s,a) = Q(s,a) + α[r + γ*Q(s',a') - Q(s,a)]
        // action_taken2 is the next action selected for the current state
        Qvalues[prev_x_pos][prev_y_pos][action_taken] = Qvalues[prev_x_pos][prev_y_pos][action_taken] +
                                                        learn_rate * (reward[x_pos][y_pos] + disc_factor * Qvalues[x_pos][y_pos][action_taken2] - Qvalues[prev_x_pos][prev_y_pos][action_taken]);
    }
    else // Update the Q value of the previous state and action if the agent has reached a terminal state
    {
        // Terminal state: Q(s,a) = Q(s,a) + α[r - Q(s,a)]
        Qvalues[prev_x_pos][prev_y_pos][action_taken] = Qvalues[prev_x_pos][prev_y_pos][action_taken] +
                                                        learn_rate * (reward[x_pos][y_pos] - Qvalues[prev_x_pos][prev_y_pos][action_taken]);
    }
}

void Qlearning()
{
    // Follow the  steps in the pseudocode in the slides
    action_taken = action_selection();
    move(action_taken);
    cum_reward = cum_reward + reward[x_pos][y_pos]; // Add the reward obtained by the agent to the cummulative reward of the agent in the current episode
    update_q_prev_state();
}

void Sarsa()
{
    move(action_taken);
    cum_reward = cum_reward + reward[x_pos][y_pos]; // Add the reward obtained by the agent to the cummulative reward of the agent in the current episode
    action_taken2 = action_selection();             // Select next action for SARSA update
    update_q_prev_state_sarsa();
    action_taken = action_taken2; // Current action becomes previous action for next iteration
}

void Multi_print_grid()
{
    int x, y;

    for (y = (height_grid - 1); y >= 0; --y)
    {
        for (x = 0; x < width_grid; ++x)
        {

            if (blocked[x][y] == 1)
            {
                cout << " \033[42m# \033[0m";
            }
            else
            {
                if ((x_pos == x) && (y_pos == y))
                {
                    cout << " \033[44m1 \033[0m";
                }
                else
                {
                    cout << " \033[31m0 \033[0m";
                }
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    reward_output.open("Rewards.txt", ios_base::app);
    Initialize_environment(); // Initialize the features of the chosen environment (goal and initial position, obstacles, rewards)

    for (i = 0; i < num_episodes; i++)
    {
        cout << "\n \n Episode " << i;
        current_episode = i;
        x_pos = init_x_pos;
        y_pos = init_y_pos;
        cum_reward = 0;

        // If Sarsa was chosen as the algorithm:
        if (algorithm == 2)
        {
            action_taken = action_selection();
        }

        // While the agent has not reached a terminal state:
        while (!(((x_pos == goalx) && (y_pos == goaly)) || ((environment == 1) && (x_pos == goalx) && (y_pos == (goaly - 1))) || ((environment == 2) && (x_pos > 0) && (x_pos < goalx) && (y_pos == 0))))
        {
            if (algorithm == 1)
            {

                Qlearning();
            }
            if (algorithm == 2)
            {
                Sarsa();
            }
        }

        finalrw[i] = cum_reward;
        cout << " Total reward obtained: " << finalrw[i] << "\n";
        reward_output << i << "," << finalrw[i] << "\n";
    }
    reward_output.close();

    return 0;
}
