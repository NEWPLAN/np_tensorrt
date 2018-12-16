cmake_minimum_required(VERSION 2.8)
project(trtexe)

set(CMAKE_VERBOSE_MAKEFILE off )
set(privateLINK "-Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group")

add_compile_options(-std=c++11 -Wall -Os)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(include)
include_directories(/usr/local/cuda/include)
include_directories(/home/newplan/software/tensorrt/TensorRT-4.0.1.6/include)

link_directories(/home/newplan/software/tensorrt/TensorRT-4.0.1.6/lib)
link_directories(/usr/local/cuda/lib64)
add_subdirectory(src)
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(trtexe main/client.cpp)

add_executable(tcp_client tcp-client/client.c)


target_link_libraries(trtexe  _3dlibs ${privateLINK} ${OpenCV_LIBS} glog)

SET(CTEST_OUTPUT_ON_FAILURE TRUE)
SET(CTEST_OUTPUT_ON_FAILURE ON)
ADD_TEST(NAME mytest
COMMAND ./trtexe --output=prob  --batch=32 --iterations=20 --deploy=../model/alexnet_deploy.prototxt)
ENABLE_TESTING()

#./trtexe --output=prob  --batch=32 --iterations=20 --deploy=../model/alexnet_deploy.prototxt --verbose
#make test ARGS="-V"