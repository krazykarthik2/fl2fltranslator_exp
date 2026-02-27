fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}
