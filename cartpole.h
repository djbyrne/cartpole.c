#ifndef CARTPOLE_H
#define CARTPOLE_H

#define OBSERVATION_SIZE 4

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
    int max_steps;

    // State variables
    double x;         // Cart position
    double x_dot;     // Cart velocity
    double theta;     // Pole angle
    double theta_dot; // Pole angular velocity

    // Other variables
    int steps_beyond_terminated;
    int current_step;
} CartPoleEnv;

typedef struct {
    double observation[OBSERVATION_SIZE]; // Observation [x, x_dot, theta, theta_dot]
    double reward;                        // Reward value
    int terminated;                       // Terminated flag
    int truncated;                        // Truncated flag
    void *info;                           // Optional info dictionary
} StepResult;

// Function declarations
void initialize(CartPoleEnv *env);
void reset(CartPoleEnv *env);
StepResult step(CartPoleEnv *env, int action);

#endif