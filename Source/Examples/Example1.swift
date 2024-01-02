import Mlx

@main
struct Example1 {
    static func main() {
        let data : [Int32] = [2, 4, 6, 8]
        let arr = Array(data, [2, 2])
        print(arr)
        print(arr.dtype())
        print(arr.shape())
        print(arr.asType(DType.int64))
    }
}
