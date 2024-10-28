#include <stdio.h>


#define INPUT_NUM 3
#define HIDDEN_NUM 1
#include "deep.h"


int main() {
  Input input = {1.8565, 1.8463, 1.8353 };
  Network network;
  init_random_weights(&network);
  float output = run_network(&input, &network);

  printf("%g\n", output);
}
