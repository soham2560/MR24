cmake_minimum_required(VERSION 3.5)

project(assignment3)

find_package(Eigen3 REQUIRED )
find_package(Ceres 2.1.0 REQUIRED)    

include_directories(
    ${EIGEN3_INCLUDE_DIR}
)
add_executable(task1 task1.cc)
target_link_libraries(
  task1
  Ceres::ceres
)