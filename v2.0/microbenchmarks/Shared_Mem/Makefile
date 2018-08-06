OBJ = o

ARCH = 35

NAME = shared_mem

$(NAME): $(NAME).$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda

$(NAME).$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -c $(INCLUDES) $<

clean:
	rm -f $(NAME).$(OBJ) $(NAME)
