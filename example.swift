import Cmlx

var data : [Float] = [2, 4, 6, 8]
var shape : [Int32] = [2, 2]

var arr = Cmlx.mlx_array_from_data(&data, &shape, 2, Cmlx.MLX_FLOAT32)
var str = String(cString: Cmlx.mlx_tostring(UnsafeMutableRawPointer(arr)))

print(str)

Cmlx.mlx_free(UnsafeMutableRawPointer(arr))
