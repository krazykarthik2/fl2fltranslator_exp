fn count_zeros(arr: *mut i32, n: i32) -> i32 {
    let mut c: i32 = 0;
    let mut i: i32 = 0;
while (i < n) { if (arr[i as usize] == 0) { let _v = c; c += 1; _v };
    { let _v = i; i += 1; _v };
}
    return c;
}
