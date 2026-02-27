fn count_down(arr: *mut i32, n: i32) {
    let mut i: i32 = 0;
while (i < n) { arr[i as usize] = (n - i);
    { let _v = i; i += 1; _v };
}
}
