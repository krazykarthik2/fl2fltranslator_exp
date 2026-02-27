fn linear_search(arr: *mut i32, n: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
while (i < n) { if (arr[i as usize] == target) { return i; }
    { let _v = i; i += 1; _v };
}
    return (-1);
}
