OBJ = o

ARCH = 35

NAME = special_function

all: $(NAME) $(NAME)_fast_math

$(NAME)_fast_math: $(NAME)_fast_math.$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda --use_fast_math

$(NAME)_fast_math.$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -c $(INCLUDES) -o $(NAME)_fast_math.$(OBJ) $< --use_fast_math

$(NAME): $(NAME).$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda

$(NAME).$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -c $(INCLUDES) $<

clean:
	rm -f $(NAME).$(OBJ) $(NAME) $(NAME)_fast_math $(NAME)_fast_math.$(OBJ)
