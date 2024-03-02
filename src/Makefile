CC = gcc
CXX = g++
NVCC = nvcc

SRCS := $(wildcard */*.cpp)
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

INCLUDES	= -I /usr/include -I include -I /usr/local/include
#-lgeos_c  
LIBS		= -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib -lboost_program_options -lpthread
CPPFLAGS	= -g -std=c++14 -Wno-cpp -fopenmp
NVCCFLAGS   = -g -std=c++14 -DUSE_GPU

GEOMETRY_SRCS := $(wildcard geometry/*.cpp)
GEOMETRY_SRCS += $(wildcard geometry/triangulate/*.cpp)
GEOMETRY_SRCS += $(wildcard storage/*.cpp)
GEOMETRY_OBJS := $(patsubst %.cpp,%.o,$(GEOMETRY_SRCS))

CU_SRCS := $(wildcard */*.cu)
CU_OBJS := $(patsubst %.cu,%_cu.o,$(CU_SRCS))

CPPFLAGS += -DUSE_GPU
LIBS  += -L/usr/local/cuda-12.2/lib64/stubs -lcuda -L/usr/local/cuda-12.2/lib64/ -lcudart
GEOMETRY_OBJS += ${CU_OBJS}


%_cu.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<	
%.o:	%.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDES) -o $@ $<	

all:	contain_polygon within_polygon


# for macro queries


contain:	query/contain_cu.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 
print_raster:	test/print_raster.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

# contain:	query/contain.o $(GEOMETRY_OBJS)
# 	$(CXX) -o ../build/$@ $^ $(LIBS) 

contain_polygon:	query/contain_polygon.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

distance:	query/distance.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

distance_polygon:	query/distance_polygon.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

debug_contain:	test/debug_contain.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS) 

debug_distance_polygon: test/debug_distance_polygon.o $(GEOMETRY_OBJS)
	$(CXX) -o ../build/$@ $^ $(LIBS)

clean:
	rm -fr ../build/* $(OBJS) $(CU_OBJS)