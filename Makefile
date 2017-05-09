LIBS = -lz

% : %.cu
	nvcc -o $* $< $(LIBS)

TARGETS = Benchmark

all : $(TARGETS)

clean :
	rm -f $(TARGETS)
