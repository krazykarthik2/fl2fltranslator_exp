fn gcd(a: i32, b: i32) -> i32 {
    while (b != 0) {
    let mut t: i32 = b;
    b = (a % b);
    a = t;
}
    return a;
}
