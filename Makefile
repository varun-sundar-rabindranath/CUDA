vecadd: vecadd.cu
	nvcc -o vecadd -ccbin g++ vecadd.cu -lcudart
