fn is_sorted(arr: *mut i32, n: i32) -> i32 {
    let mut i: i32 = 0;
while (i < (n - 1)) { if (arr[i as usize] > arr[(i + 1) as usize]) { return 0; }
    { let _v = i; i += 1; _v };
}
    return 1;
}
