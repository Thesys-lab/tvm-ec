target_files = simple_gemm

simple-gemm:
	icc -qmkl -o simple_gemm src/simple_gemm.c

run:
	./simple_gemm

clean:
	rm -f $(target_files)