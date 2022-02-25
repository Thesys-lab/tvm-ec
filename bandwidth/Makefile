target_files = stream stream_openmp stream_icc

stream:
	gcc -O stream.c -o stream

stream-openmp:
	gcc -O -DNTIMES=100 -DOFFSET=0 -DSTREAM_TYPE=double -DSTREAM_ARRAY_SIZE=268435456 -Wall -O3 -mcmodel=medium -fopenmp stream.c -o stream_openmp

stream-icc:
	icc -DNTIMES=100 -DOFFSET=0 -DSTREAM_TYPE=double -DSTREAM_ARRAY_SIZE=268435456 -Wall -O3 -mcmodel=medium -qopenmp -shared-intel -qopt-streaming-stores always stream.c -o stream_icc

clean:
	rm -f $(target_files)