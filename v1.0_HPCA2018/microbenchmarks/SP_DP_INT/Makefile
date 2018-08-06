OBJ = o

ARCH = 35

NAME = computeKernel

$(NAME): $(NAME).$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda

$(NAME).$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -c $(INCLUDES) $<

clean:
	rm -f $(NAME).$(OBJ) $(NAME)
