fn dot_product(a: *mut i32, b: *mut i32, n: i32) -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
while (i < n) { s += (a[i as usize] * b[i as usize]);
    { let _v = i; i += 1; _v };
}
    return s;
}
