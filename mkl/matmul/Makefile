target_files = simple_gemm

simple-gemm:
	icc -qmkl -o simple_gemm src/simple_gemm.c

run:
	./simple_gemm $(m) $(n) $(k)

clean:
	rm -f $(target_files)