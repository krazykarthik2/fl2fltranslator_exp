fn clear_bit(x: i32, bit: i32) -> i32 {
    return (x & ~(1 << bit));
}
