fn increment_all(p: *mut i32, n: i32) {
    let mut i: i32 = 0;
while (i < n) { { let _v = p[i as usize]; p[i as usize] += 1; _v };
    { let _v = i; i += 1; _v };
}
}
