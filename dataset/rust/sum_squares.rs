fn sum_squares(n: i32) -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 1;
while (i <= n) { s += (i * i);
    { let _v = i; i += 1; _v };
}
    return s;
}
