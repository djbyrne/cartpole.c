// agent.c

#include <stdlib.h>
#include "agent.h"

// Simple agent that selects actions randomly
int agent_policy(double state[4]) {
    // For now, ignore the state and return a random action (0 or 1)
    int action = rand() % 2;
    return action;
}