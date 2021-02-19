TARGET		= cuMemAllocTest

SOURCES		= cuMemAllocTest.cu

OBJECTS		= $(SOURCES:.cu=.o)

CC		= mpicc
CPPFLAGS	+= -DHAVE_MPI
CFLAGS		+= 
LDFLAGS		+=
LIBS		+= -lm -lcudart -lcuda

NVCC		= nvcc --forward-unknown-to-host-compiler

##

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean::
	$(RM) $(TARGET) $(OBJECTS)

%.o: %.cu
	$(NVCC) -o $@ -c -ccbin $(CC) $(CPPFLAGS) $(CFLAGS) $^

