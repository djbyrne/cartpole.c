// main.c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cartpole.h"
#include "agent.h"

int main() {
    CartPoleEnv env;
    initialize(&env);

    int done = 0;
    double reward = 0.0;
    int time_steps = 0;
    const int max_time_steps = 500;

    // Seed the random number generator
    srand(time(NULL));

    // Array to hold the state
    double state[4];

    // Reset the environment
    reset(&env);

    while (!done && time_steps < max_time_steps) {
        // Get the current state
        state[0] = env.x;
        state[1] = env.x_dot;
        state[2] = env.theta;
        state[3] = env.theta_dot;

        // Agent selects an action based on the current state
        int action = agent_policy(state);

        // Environment takes a step based on the action
        done = step(&env, action, &reward);

        // Print out information
        printf("Time Step: %d, Action: %d, Reward: %.2f, Done: %d\n", time_steps, action, reward, done);

        time_steps++;
    }

    // Episode ended
    if (done) {
        printf("Episode terminated after %d time steps.\n", time_steps);
    } else {
        printf("Episode reached max time steps (%d).\n", max_time_steps);
    }

    return 0;
}