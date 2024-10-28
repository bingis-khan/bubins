build: prog
prog: src/main.c src/deep.h
	cc src/main.c -o prog -g -L./ext -lraylib -framework Cocoa -framework OpenGL -framework IOKit

run: prog
	./prog data/capacities.csv

test: src/test.c src/deep.h
	cc src/test.c -o test -g

.PHONY: clean
clean:
	rm -f prog
