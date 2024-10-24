#include <stdio.h>
#include <stdlib.h>

#include "raylib.h"


#define MAX_LINE_LEN 256

#define CAPACITY_LEN 168
float capacities[CAPACITY_LEN];


#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600


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



  // Window stuff begin
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "le epic machine learning");

  while (!WindowShouldClose()) {
    BeginDrawing();
      ClearBackground(RAYWHITE);

      #define WINDOW_OFFSET 20
      #define POINT_SIZE 7
      #define POINT_X_OFFSET ((WINDOW_WIDTH - 2*WINDOW_OFFSET) / CAPACITY_LEN)
      #define POINT_Y_SCALE ((WINDOW_HEIGHT - 2*WINDOW_OFFSET))

      for (int i = 0; i < CAPACITY_LEN; i++) {
        float capacity = capacities[i];
        float norm_cap = 1 - (capacity - 1);
        float graph_y = norm_cap * POINT_Y_SCALE + WINDOW_OFFSET;

        DrawCircle(i * POINT_X_OFFSET + WINDOW_OFFSET, graph_y, POINT_SIZE, RED);
        
      }
    EndDrawing();
  }

  CloseWindow();
}
