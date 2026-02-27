fn min_of_three(a: i32, b: i32, c: i32) -> i32 {
    let mut m: i32 = a;
    if (b < m) { m = b; }
    if (c < m) { m = c; }
    return m;
}
