OBJ = o

ARCH = 35
CUDA_FLAG = -arch sm_$(ARCH)

KERNEL_FILE = cf_kernel
KERNEL_SHIFT_FILE = cf_shift_kernel

NAME = cf
NAME_SHIFT = cf_shift

all: $(NAME) $(NAME_SHIFT)
kernels: $(KERNEL_FILE).$(OBJ) $(KERNEL_SHIFT_FILE).$(OBJ)

#BINS
$(NAME): $(KERNEL_FILE).$(OBJ) $(NAME).$(OBJ)
	nvcc $(CUDA_FLAG) -o $@ $^ -lcuda -lcudart

$(NAME_SHIFT): $(KERNEL_SHIFT_FILE).$(OBJ) $(NAME_SHIFT).$(OBJ)
	nvcc $(CUDA_FLAG) -o $@ $^ -lcuda -lcudart

#OBJS
$(NAME).$(OBJ): $(NAME).cu
	nvcc $(CUDA_FLAG) -c $^ -o $@ -lcuda -lcudart

$(NAME_SHIFT).$(OBJ): $(NAME_SHIFT).cu
	nvcc $(CUDA_FLAG) -c $^ -o $@ -lcuda -lcudart

#KERNELS
$(KERNEL_FILE).$(OBJ):	$(KERNEL_FILE).cu
	nvcc $(CUDA_FLAG) -c $^ -o $@ -lcuda -lcudart

$(KERNEL_SHIFT_FILE).$(OBJ):	$(KERNEL_SHIFT_FILE).cu
	nvcc $(CUDA_FLAG) -c $^ -o $@ -lcuda -lcudart

clean:
	rm -f *.$(OBJ) $(NAME) $(NAME_SHIFT)
