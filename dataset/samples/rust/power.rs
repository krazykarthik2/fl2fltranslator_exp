fn power(base: i64, exp: i32) -> i64 {
    let mut r: i64 = 1;
    while (exp > 0) {
    r *= base;
    { let _v = exp; exp -= 1; _v };
}
    return r;
}
