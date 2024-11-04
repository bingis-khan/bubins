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
#ifndef HIDDEN2_NUM
  #error "You must define HIDDEN_NUM (number of nodes in the hidden layer)"

  // define a placeholder to silence other errors
  #define HIDDEN2_NUM 20
#endif

typedef float Input[INPUT_NUM];
typedef float Hidden[HIDDEN_NUM];
typedef float Hidden2[HIDDEN_NUM];
typedef float HiddenWeights[HIDDEN_NUM][INPUT_NUM];
typedef float Hidden2Weights[HIDDEN2_NUM][HIDDEN_NUM];
typedef float OutputWeights[HIDDEN2_NUM];
typedef struct {
  HiddenWeights hidden_weights;
  Hidden2Weights hidden2_weights;
  OutputWeights output_weights;
} Network;

void init_random_weights(Network *weights);
float run_network(Input *input, Network *network);
void backprop(Input *input, float expected, Network *network);


// -------------- IMPLEMENTATION --------------

static float relu(float);
static float relu_deriv(float);
static float leaky_relu(float);
static float leaky_relu_deriv(float);

static float random_weight() {
  return ((rand() / (float)RAND_MAX) * .2);
}

void init_random_weights(Network *network) {
  for (int i = 0; i < INPUT_NUM; i++) {
    for (int w = 0; w < HIDDEN_NUM; w++) {
      network->hidden_weights[w][i] = random_weight();
    }
  }

  for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
    for (int w = 0; w < HIDDEN_NUM; w++) {
      network->hidden2_weights[w2][w] = random_weight();
    }
  }

  for (int w = 0; w < HIDDEN2_NUM; w++) {
    network->output_weights[w] = random_weight();
  }
}

float run_network(Input *input, Network *network) {
  Hidden hidden = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int i = 0; i < INPUT_NUM; i++) {
      float weighed_input = leaky_relu((*input)[i] * network->hidden_weights[w][i]);
      hidden[w] += weighed_input;
    }
  }

  Hidden2 hidden2 = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
      float weighed_hidden = leaky_relu(hidden[w] * network->hidden2_weights[w2][w]);
      hidden2[w2] += weighed_hidden;
    }
  }
  
  float output = 0.;
  for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
    output += hidden2[w2] * network->output_weights[w2];
  }

  return output;
}

void backprop(Input *input, float expected, Network *network) {
  // forward
  Hidden hidden = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int i = 0; i < INPUT_NUM; i++) {
      float weighed_input = leaky_relu((*input)[i] * network->hidden_weights[w][i]);
      hidden[w] += weighed_input;
    }
  }

  Hidden2 hidden2 = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
      float weighed_hidden = leaky_relu(hidden[w] * network->hidden2_weights[w2][w]);
      hidden2[w2] += weighed_hidden;
    }
  }
  
  float output = 0.;
  for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
    output += hidden2[w2] * network->output_weights[w2];
  }


  // ----- backprop -----
  
  // output layer delta
  float out_delta = expected - output;

  // hidden2 layer delta
  Hidden2 hidden2_delta = { 0 };
  for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
    hidden2_delta[w2] = network->output_weights[w2] * out_delta * leaky_relu_deriv(hidden2[w2]);
  }

  // hidden layer delta
  Hidden hidden_delta = { 0 };
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
      hidden_delta[w] += network->hidden2_weights[w2][w] * hidden2_delta[w2] * leaky_relu_deriv(hidden[w]);
    }
  }

  // update hid-out weights
  #define ALPHA .05
  for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
    network->output_weights[w2] += ALPHA * hidden2[w2] * out_delta;
  }

  // update in-hid weights
  for (int w = 0; w < HIDDEN_NUM; w++) {
    for (int w2 = 0; w2 < HIDDEN2_NUM; w2++) {
      network->hidden2_weights[w2][w] += ALPHA * hidden[w] * hidden2_delta[w2];
    }
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

static inline float leaky_relu(float value) {
  return value < 0
    ? .01 * value
    : value;
}

static inline float leaky_relu_deriv(float value) {
  return value > 0
    ? 1
    : .01;
}

#endif
