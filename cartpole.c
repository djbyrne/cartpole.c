#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cartpole.h"


#define MAX_STEPS 500 // Maximum steps per episode

void initialize(CartPoleEnv *env) {
    // Seed the random number generator
    srand(42);

    // Environment parameters
    env->gravity = 9.8;
    env->mass_cart = 1.0;
    env->mass_pole = 0.1;
    env->total_mass = env->mass_pole + env->mass_cart;
    env->length = 0.5; // Half the pole's length
    env->polemass_length = env->mass_pole * env->length;
    env->force_mag = 10.0;
    env->tau = 0.02; // Time interval for updates
    env->theta_threshold_radians = 12 * 2 * M_PI / 360;
    env->x_threshold = 2.4;
    env->max_steps = MAX_STEPS; // Maximum number of steps

    // Reset the environment state
    reset(env);
}

void reset(CartPoleEnv *env) {
    // Randomly initialize state variables in (-0.05, 0.05)
    env->x = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    env->x_dot = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    env->theta = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    env->theta_dot = ((double)rand() / RAND_MAX) * 0.1 - 0.05;

    env->steps_beyond_terminated = -1;
    env->current_step = 0; // Reset step counter
}

StepResult step(CartPoleEnv *env, int action) {
    StepResult result;
    
    // Validate action
    if (action != 0 && action != 1) {
        printf("Invalid action: %d\n", action);
        result.terminated = -1;
        return result;
    }

    // Unpack state variables
    double x = env->x;
    double x_dot = env->x_dot;
    double theta = env->theta;
    double theta_dot = env->theta_dot;

    // Determine force based on action
    double force = action == 1 ? env->force_mag : -env->force_mag;
    double costheta = cos(theta);
    double sintheta = sin(theta);

    // Compute acceleration
    double temp = (force + env->polemass_length * theta_dot * theta_dot * sintheta) / env->total_mass;
    double thetaacc = (env->gravity * sintheta - costheta * temp) /
                      (env->length * (4.0 / 3.0 - env->mass_pole * costheta * costheta / env->total_mass));
    double xacc = temp - env->polemass_length * thetaacc * costheta / env->total_mass;

    // Update state variables
    x += env->tau * x_dot;
    x_dot += env->tau * xacc;
    theta += env->tau * theta_dot;
    theta_dot += env->tau * thetaacc;

    env->x = x;
    env->x_dot = x_dot;
    env->theta = theta;
    env->theta_dot = theta_dot;

    // Increment step count
    env->current_step++;

    // Check if terminated
    int terminated = x < -env->x_threshold || x > env->x_threshold ||
                     theta < -env->theta_threshold_radians || theta > env->theta_threshold_radians;

    // Check if truncated (max steps reached)
    int truncated = env->current_step >= env->max_steps;

    // Set reward
    if (!terminated && !truncated) {
        result.reward = 1.0;
    } else if (terminated && env->steps_beyond_terminated == -1) {
        // Pole just fell
        env->steps_beyond_terminated = 0;
        result.reward = 1.0;
    } else {
        if (env->steps_beyond_terminated == 0) {
            printf("Warning: Step called after termination. Reset environment.\n");
        }
        env->steps_beyond_terminated++;
        result.reward = 0.0;
    }

    // Populate StepResult struct
    result.observation[0] = env->x;
    result.observation[1] = env->x_dot;
    result.observation[2] = env->theta;
    result.observation[3] = env->theta_dot;
    result.terminated = terminated;
    result.truncated = truncated;
    //add info dictionary here, cant be NULL
    result.info = malloc(sizeof(int));

    return result;
}

// Main function

int main() {
    return 0;
}