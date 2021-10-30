PREFIX = /usr/local/cuda/bin
CXX = $(PREFIX)/nvcc
CXX_FLAGS = -g -O3
CXX_LIBS = 
BIN = md5_gpu

main:
	$(CXX) $(CXX_FLAGS) md5_gpu.cu -o $(BIN) $(CXX_LIBS)
