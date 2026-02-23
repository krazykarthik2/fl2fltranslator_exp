fn find_max(arr: *mut i32, n: i32) -> i32 {
    let mut m: i32 = arr[0 as usize];
    let mut i: i32 = 1;
while (i < n) { if (arr[i as usize] > m) { m = arr[i as usize]; }
    { let _v = i; i += 1; _v };
}
    return m;
}
