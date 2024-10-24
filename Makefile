build: prog
prog: src/main.c
	cc src/main.c -o prog -L./ext -lraylib -framework Cocoa -framework OpenGL -framework IOKit

run: prog
	./prog data/capacities.csv

.PHONY: clean
clean:
	rm -f prog
