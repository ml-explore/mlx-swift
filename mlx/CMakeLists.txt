add_library(Mlx STATIC array.swift device.swift stream.swift)
target_include_directories(Mlx PUBLIC "${CMAKE_SOURCE_DIR}/Cmlx" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(Mlx PRIVATE mlxc)
