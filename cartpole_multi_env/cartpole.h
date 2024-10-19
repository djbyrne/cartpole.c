// cartpole.h

#ifndef CARTPOLE_H
#define CARTPOLE_H

#include <stddef.h>

typedef struct {
    // Environment parameters
    double gravity;
    double mass_cart;
    double mass_pole;
    double total_mass;
    double length; // Half the pole's length
    double polemass_length;
    double force_mag;
    double tau; // Time interval for updates
    double theta_threshold_radians;
    double x_threshold;

    // State variables (arrays)
    size_t num_envs;     // Number of environments
    double *x;           // Cart positions
    double *x_dot;       // Cart velocities
    double *theta;       // Pole angles
    double *theta_dot;   // Pole angular velocities

    // Done flags
    int *done;           // Flags indicating if an environment is done
} CartPoleEnvBatch;

// Function declarations
CartPoleEnvBatch* create_env_batch(size_t num_envs);
void free_env_batch(CartPoleEnvBatch *batch);
void reset_env_batch(CartPoleEnvBatch *batch);
void step_env_batch(CartPoleEnvBatch *batch, int *actions, double *rewards);

#endif