.PHONY: simple-dgemm simple-sgemm all run clean

target_files = simple_dgemm simple_sgemm


simple-dgemm:
	icc -qmkl -o simple_dgemm src/simple_dgemm.c

simple-sgemm:
	icc -qmkl -o simple_sgemm src/simple_sgemm.c

all: simple-dgemm simple-sgemm

run:
ifeq ($(dtype),d)
	./simple_dgemm $(m) $(n) $(k)
else
	./simple_sgemm $(m) $(n) $(k)
endif

clean:
	rm -f $(target_files)