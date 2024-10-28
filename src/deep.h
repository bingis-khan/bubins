#ifndef DEEP_H
#define DEEP_H

#include <stdlib.h>


#ifndef INPUT_NUM
  #error "You must define INPUT_NUM (number of inputs to the neural network)"

  // define a placeholder to silence other errors
  #define INPUT_NUM 10
#endif
#ifndef HIDDEN_NUM
  #error "You must define HIDDEN_NUM (number of nodes in the hidden layer)"

  // define a placeholder to silence other errors
  #define HIDDEN_NUM 20
#endif

typedef float Input[INPUT_NUM];
typedef float Hidden[HIDDEN_NUM];
typedef float HiddenWeights[HIDDEN_NUM][INPUT_NUM];
typedef float OutputWeights[HIDDEN_NUM];
typedef struct {
  HiddenWeights hidden_weights;
  OutputWeights output_weights;
} Network;

void init_random_weights(Network *weights);
float run_network(Input *input, Network *network);
void backprop(Input *input, float expected, Network *network);


// -------------- IMPLEMENTATION --------------

static float relu(float);
static float relu_deriv(float);

void init_random_weights(Network *network) {
  for (int i = 0; i < INPUT_NUM; i++) {
    for (int w = 0; w < HIDDEN_NUM; w++) {
      network->hidden_weights[w][i] = ((rand() / (float)RAND_MAX) * 2.) - 1.;
    }
  }

  for (int w = 0; w < HIDDEN_NUM; w++) {
    network->output_weights[w] = ((rand() / (float)RAND_MAX) * 2.) - 1.;
  }
}

float run_network(Input *input, Network *network) {
  Hidden hidden = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int i = 0; i < INPUT_NUM; i++) {
      float weighed_input = relu((*input)[i] * network->hidden_weights[w][i]);
      hidden[w] += weighed_input;
    }
  }

  float output = 0.;
  for (int w = 0; w < HIDDEN_NUM; w++) {
    float weighed_out = hidden[w] * network->output_weights[w];
    output += weighed_out;
  }
  
  return output;
}

void backprop(Input *input, float expected, Network *network) {
  // forward
  Hidden hidden = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int i = 0; i < INPUT_NUM; i++) {
      hidden[w] += relu((*input)[i] * network->hidden_weights[w][i]);
    }
  }

  float output = 0.;
  for (int w = 0; w < HIDDEN_NUM; w++) {
    output += hidden[w] * network->output_weights[w];
  }

  // ----- backprop -----
  
  // output layer delta
  float out_delta = expected - output;

  // hidden layer delta
  Hidden hidden_delta = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    hidden_delta[w] = network->output_weights[w] * out_delta * relu_deriv(hidden[w]);
  }

  // update hid-out weights
  #define ALPHA .2
  for (int w = 0; w < HIDDEN_NUM; w++) {
    network->output_weights[w] += ALPHA * hidden[w] * out_delta;
  }

  // update in-hid weights
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int i = 0; i < INPUT_NUM; i++) {
      network->hidden_weights[w][i] += ALPHA * (*input)[i] * hidden_delta[w];
    }
  }
}

static inline float relu(float value) {
  return value < 0
    ? 0
    : value;
}

static inline float relu_deriv(float value) {
  return value > 0
    ? 1
    : 0;
}

#endif
