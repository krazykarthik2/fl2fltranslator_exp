fn fill_array(arr: *mut i32, n: i32, val: i32) {
    let mut i: i32 = 0;
while (i < n) { arr[i as usize] = val;
    { let _v = i; i += 1; _v };
}
}
