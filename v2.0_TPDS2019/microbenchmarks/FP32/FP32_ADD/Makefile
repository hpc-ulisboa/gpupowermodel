OBJ = o

ARCH = 35

NAME = fp32_add

$(NAME): $(NAME).$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda

$(NAME).$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -Xptxas -O0,-dlcm=cv,-dscm=wt  -c $(INCLUDES) $<

clean:
	rm -f $(NAME).$(OBJ) $(NAME)
