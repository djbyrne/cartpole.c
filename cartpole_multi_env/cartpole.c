// cartpole.c

#include "cartpole.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

CartPoleEnvBatch* create_env_batch(size_t num_envs) {
    CartPoleEnvBatch *batch = (CartPoleEnvBatch *)malloc(sizeof(CartPoleEnvBatch));
    if (!batch) {
        return NULL;
    }

    // Initialize environment parameters
    batch->gravity = 9.8;
    batch->mass_cart = 1.0;
    batch->mass_pole = 0.1;
    batch->total_mass = batch->mass_pole + batch->mass_cart;
    batch->length = 0.5; // Half the pole's length
    batch->polemass_length = batch->mass_pole * batch->length;
    batch->force_mag = 10.0;
    batch->tau = 0.02; // Time interval for updates
    batch->theta_threshold_radians = 12 * 2 * M_PI / 360;
    batch->x_threshold = 2.4;

    // Allocate arrays for state variables
    batch->num_envs = num_envs;
    batch->x = (double *)malloc(sizeof(double) * num_envs);
    batch->x_dot = (double *)malloc(sizeof(double) * num_envs);
    batch->theta = (double *)malloc(sizeof(double) * num_envs);
    batch->theta_dot = (double *)malloc(sizeof(double) * num_envs);
    batch->done = (int *)malloc(sizeof(int) * num_envs);

    if (!batch->x || !batch->x_dot || !batch->theta || !batch->theta_dot || !batch->done) {
        free_env_batch(batch);
        return NULL;
    }

    // Initialize done flags
    memset(batch->done, 0, sizeof(int) * num_envs);

    return batch;
}

void free_env_batch(CartPoleEnvBatch *batch) {
    if (batch) {
        free(batch->x);
        free(batch->x_dot);
        free(batch->theta);
        free(batch->theta_dot);
        free(batch->done);
        free(batch);
    }
}

void reset_env_batch(CartPoleEnvBatch *batch) {
    for (size_t i = 0; i < batch->num_envs; i++) {
        // Randomly initialize state variables in (-0.05, 0.05)
        batch->x[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        batch->x_dot[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        batch->theta[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        batch->theta_dot[i] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
        batch->done[i] = 0;
    }
}
void step_env_batch(CartPoleEnvBatch *batch, int *actions, double *rewards) {
    size_t num_envs = batch->num_envs;

    #pragma omp parallel for
    for (size_t i = 0; i < num_envs; i++) {
        if (batch->done[i]) {
            rewards[i] = 0.0;
            continue;
        }

        // Unpack state variables
        double x = batch->x[i];
        double x_dot = batch->x_dot[i];
        double theta = batch->theta[i];
        double theta_dot = batch->theta_dot[i];

        // Validate action
        int action = actions[i];
        if (action != 0 && action != 1) {
            // printf("Invalid action: %d\n", action);
            batch->done[i] = 1;
            rewards[i] = 0.0;
            continue;
        }

        // Determine force based on action
        double force = action == 1 ? batch->force_mag : -batch->force_mag;
        double costheta = cos(theta);
        double sintheta = sin(theta);

        // Compute acceleration
        double temp = (force + batch->polemass_length * theta_dot * theta_dot * sintheta) / batch->total_mass;
        double thetaacc = (batch->gravity * sintheta - costheta * temp) /
                          (batch->length * (4.0 / 3.0 - batch->mass_pole * costheta * costheta / batch->total_mass));
        double xacc = temp - batch->polemass_length * thetaacc * costheta / batch->total_mass;

        // Update state variables
        x += batch->tau * x_dot;
        x_dot += batch->tau * xacc;
        theta += batch->tau * theta_dot;
        theta_dot += batch->tau * thetaacc;

        batch->x[i] = x;
        batch->x_dot[i] = x_dot;
        batch->theta[i] = theta;
        batch->theta_dot[i] = theta_dot;

        // Check if terminated
        int done = x < -batch->x_threshold || x > batch->x_threshold ||
                   theta < -batch->theta_threshold_radians || theta > batch->theta_threshold_radians;

        batch->done[i] = done;

        // Set reward
        rewards[i] = done ? 0.0 : 1.0;
    }
}