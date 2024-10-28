#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "raylib.h"

// ------ define our neural network. ------

#define INPUT_NUM 3
#define HIDDEN_NUM 2
#include "deep.h"

// -------

#define MAX_LINE_LEN 256

#define CAPACITY_LEN 168
float capacities[CAPACITY_LEN];
float learned_capacities[CAPACITY_LEN];
float casc_learned_capacities[CAPACITY_LEN];

// Store last FITNESS_LEN fitnesses, circular buffer style.
#define FITNESS_LEN 50
float fitness[FITNESS_LEN];
int fit_i = 0;
int fit_num = 0;

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

#define FIGURE_OFFSET 20
#define BOUNDING_OFFSET (FIGURE_OFFSET/2)

// should probably add a macro for normalization.


// ------- Declarations ------- 

static void graph_capacities(float *capacities, Color color, int fig_x, int fig_y, int fig_width, int fig_height);
static void graph_fitness(int fig_x, int fig_y, int fig_width, int fig_height);
static void add_fitness(float fit);
static void train_whole(Network *network);
static float point_error(Network *network);


// ------- Program start -------

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Required data file.\n");
    return 1;
  }

  // Load data file
  char* filename = argv[1];
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not open file.\n");
    return 1;
  }
  char line_buf[MAX_LINE_LEN];

  for (int i = 0; fgets(line_buf, sizeof(line_buf), file) != NULL; i++) {
    float capacity = atof(line_buf);
    capacities[i] = capacity;
  }

  fclose(file);

  // Initialize neural network.
  Network network;
  init_random_weights(&network);


  // Window stuff begin
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "le epic machine learning");

  SetTargetFPS(60);
  while (!WindowShouldClose()) {
    train_whole(&network);

    // calculate mse (to check fitness)
    Input input;
    for (int ii = 0; ii < INPUT_NUM; ii++) {
      input[ii] = capacities[ii];
    }

    float mse = point_error(&network);
    add_fitness(mse);

    // graph learned capacities.
    for (int i = 0; i < INPUT_NUM; i++) {
      learned_capacities[i] = capacities[i];
    }

    for (int i = INPUT_NUM; i < CAPACITY_LEN; i++) {
      Input input;
      for (int ii = 0; ii < INPUT_NUM; ii++) {
        input[ii] = capacities[i + ii - INPUT_NUM];
      }

      float output = run_network(&input, &network);
      learned_capacities[i] = output;
    }

    // graph learned capacities. (cascading)
    for (int i = 0; i < INPUT_NUM; i++) {
      casc_learned_capacities[i] = capacities[i];
    }

    for (int i = INPUT_NUM; i < CAPACITY_LEN; i++) {
      Input input;
      for (int ii = 0; ii < INPUT_NUM; ii++) {
        input[ii] = casc_learned_capacities[i + ii - INPUT_NUM];
      }

      float output = run_network(&input, &network);
      casc_learned_capacities[i] = output;
    }


    BeginDrawing();
      ClearBackground(RAYWHITE);
      graph_capacities(capacities, RED, 0, 0, WINDOW_WIDTH/2, WINDOW_HEIGHT/2);
      graph_capacities(learned_capacities, BLUE, 0, 0, WINDOW_WIDTH/2, WINDOW_HEIGHT/2);
      graph_capacities(casc_learned_capacities, GREEN, 0, 0, WINDOW_WIDTH/2, WINDOW_HEIGHT/2);
      graph_fitness(0, WINDOW_HEIGHT/2, WINDOW_WIDTH, WINDOW_HEIGHT/2);

      char mse_buf[20];
      snprintf(mse_buf, sizeof(mse_buf), "%g", mse);
      DrawText(mse_buf, 0, 0, 20, GREEN);
      // DrawFPS(0, 0);
    EndDrawing();
  }

  CloseWindow();
}

// Train the network based on the dataset.
static void train_whole(Network *network) {
  for (int i = INPUT_NUM; i < CAPACITY_LEN - INPUT_NUM; i++) {
    // slice into input.
    Input input;
    for (int ii = 0; ii < INPUT_NUM; ii++) {
      input[ii] = capacities[i + ii - INPUT_NUM];
    }

    float expected = capacities[i];

    backprop(&input, expected, network);
  }
}

// Calculates error (without cascading errors.)
static float point_error(Network *network) {
  float mse = 0.;
  for (int i = INPUT_NUM; i < CAPACITY_LEN - INPUT_NUM; i++) {
    // slice into input.
    Input input;
    for (int ii = 0; ii < INPUT_NUM; ii++) {
      input[ii] = capacities[ii + i - INPUT_NUM];
    }

    float output = run_network(&input, network);
    float expected = capacities[i];
    float delta = expected - output;
    mse += delta * delta;
  }

  return mse / (CAPACITY_LEN - INPUT_NUM);
}


// Add to fitness function.
static void add_fitness(float fit) {
  int next_fit = (fit_i + 1) % FITNESS_LEN;
  fitness[next_fit] = fit;
  fit_i = next_fit;
  if (fit_num < FITNESS_LEN) fit_num++;
}



// Draw graph of fitness.
static void graph_fitness(int fig_x, int fig_y, int fig_width, int fig_height) {
  #define POINT_SIZE 3
  float point_x_offset = (fig_width - 2*FIGURE_OFFSET) / (float)FITNESS_LEN;
  float point_y_scale = fig_height - 2*FIGURE_OFFSET;

  // find maximum value and normalize everything based on it.
  float max = 0.;
  for (int i = fit_i, n = 0; n < fit_num; i = i <= 0 ? FITNESS_LEN - 1 : i - 1, n++) {
    if (fitness[i] > max) max = fitness[i];
  }

  // minimum max value - graph will look better.
  if (max < 1.) max = 1.;

  for (int i = fit_i, n = 0; n < fit_num; i = i <= 0 ? FITNESS_LEN - 1 : i - 1, n++) {
    float fit = fitness[i];
    float norm_fit = 1 - (fit / max);
    float graph_y = norm_fit * point_y_scale + FIGURE_OFFSET;

    int cur_x = fig_x + point_x_offset / 2 + (FITNESS_LEN - 1 - n) * point_x_offset + FIGURE_OFFSET;
    int cur_y = fig_y + graph_y;

    if (n < fit_num - 1) {
      float next_norm_fit = 1 - (fitness[i <= 0 ? FITNESS_LEN - 1 : i - 1] / max);
      float next_graph_y = next_norm_fit * point_y_scale + FIGURE_OFFSET;

      int next_x = fig_x + point_x_offset / 2 + (FITNESS_LEN - 1 - (n + 1)) * point_x_offset + FIGURE_OFFSET;
      int next_y = fig_y + next_graph_y;

      DrawLine(cur_x, cur_y, next_x, next_y, BLACK);
    } else if (fit_num == 1) {
      DrawCircle(fig_x + point_x_offset / 2 + (FITNESS_LEN - 1 - n) * point_x_offset + FIGURE_OFFSET, fig_y + graph_y, POINT_SIZE, RED);
    }
  }

  // annotate maximum
  float max_norm_fit = 1 - (max / max);
  float line_y = fig_y + max_norm_fit * point_y_scale + FIGURE_OFFSET;
  DrawLine(fig_x, line_y, fig_x + FIGURE_OFFSET, line_y, RED);

  char max_text[20];
  snprintf(max_text, sizeof(max_text), "%g", max);
  DrawText(max_text, fig_x + FIGURE_OFFSET, line_y, 20, RED);

  if (fit_num > 0) {
    char cur_text[20];
    snprintf(cur_text, sizeof(cur_text), "%g", fitness[fit_i]);

    float cur_norm_fit = 1 - (fitness[fit_i]/ max);
    float line_y = fig_y + cur_norm_fit * point_y_scale + FIGURE_OFFSET;
    int text_offset = MeasureText(cur_text, 20);
    DrawText(cur_text, fig_x + fig_width - FIGURE_OFFSET - text_offset, line_y, 20, RED);
  }

  DrawRectangleLines(fig_x + BOUNDING_OFFSET, fig_y + BOUNDING_OFFSET, fig_width - BOUNDING_OFFSET*2, fig_height - BOUNDING_OFFSET*2, RED);
}


// Draw graph of our capacities dataset.
static void graph_capacities(float *capacities, Color color, int fig_x, int fig_y, int fig_width, int fig_height) {
  #define POINT_SIZE 3
  float point_x_offset = (fig_width - 2*FIGURE_OFFSET) / (float)CAPACITY_LEN;
  float point_y_scale = fig_height - 2*FIGURE_OFFSET;

  for (int i = 0; i < CAPACITY_LEN; i++) {
    float capacity = capacities[i];
    float norm_cap = 1 - (capacity - 1);  // normalization based on actual capacity values, which seem to be between (1, 2). then reversed, because because raylib.
    float graph_y = norm_cap * point_y_scale + FIGURE_OFFSET;

    DrawCircle(fig_x + point_x_offset / 2 + i * point_x_offset + FIGURE_OFFSET, fig_y + graph_y, POINT_SIZE, color);
  }

  DrawRectangleLines(fig_x + BOUNDING_OFFSET, fig_y + BOUNDING_OFFSET, fig_width - BOUNDING_OFFSET*2, fig_height - BOUNDING_OFFSET*2, RED);
}
