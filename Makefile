# Makefile for CUDA GEMM

# Compiler
NVCC = nvcc

# Target executable
TARGET = gemm

# Source files
SOURCES = main.cu

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_89
LIBS = -lcublas

# Matrix sizes (can be overridden)
M_SIZE ?= 4096
N_SIZE ?= 4096
K_SIZE ?= 4096

# Add size definitions if they're not default
ifneq ($(M_SIZE),4096)
	NVCC_FLAGS += -DM_SIZE=$(M_SIZE)
endif
ifneq ($(N_SIZE),4096)
	NVCC_FLAGS += -DN_SIZE=$(N_SIZE)
endif
ifneq ($(K_SIZE),4096)
	NVCC_FLAGS += -DK_SIZE=$(K_SIZE)
endif

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(LIBS) -o $(TARGET) $(SOURCES)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Rebuild
rebuild: clean all

.PHONY: all run clean rebuild
