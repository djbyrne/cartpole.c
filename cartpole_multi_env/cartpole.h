// cartpole.h

#ifndef CARTPOLE_H
#define CARTPOLE_H

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

    // State variables
    double x;         // Cart position
    double x_dot;     // Cart velocity
    double theta;     // Pole angle
    double theta_dot; // Pole angular velocity

    // Other variables
    int steps_beyond_terminated;
} CartPoleEnv;

// Function declarations
void initialize(CartPoleEnv *env);
void reset(CartPoleEnv *env);
int step(CartPoleEnv *env, int action, double *reward);

#endif