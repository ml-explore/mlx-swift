namespace mlx::core::metal {

const char* ternary_ops() {
  return R"preamble(
struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};
)preamble";
}

} // namespace mlx::core::metal
