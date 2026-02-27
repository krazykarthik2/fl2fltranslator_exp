fn sum_range(lo: i32, hi: i32) -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = lo;
while (i <= hi) { s += i;
    { let _v = i; i += 1; _v };
}
    return s;
}
