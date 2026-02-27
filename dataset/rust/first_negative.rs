fn first_negative(arr: *mut i32, n: i32) -> i32 {
    let mut i: i32 = 0;
while (i < n) { if (arr[i as usize] < 0) { return i; }
    { let _v = i; i += 1; _v };
}
    return (-1);
}
