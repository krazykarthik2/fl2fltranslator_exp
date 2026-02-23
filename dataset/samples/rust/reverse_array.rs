fn reverse_array(arr: *mut i32, n: i32) {
    let mut i: i32 = 0;
    let mut j: i32 = (n - 1);
    while (i < j) {
    let mut t: i32 = arr[i as usize];
    arr[i as usize] = arr[j as usize];
    arr[j as usize] = t;
    { let _v = i; i += 1; _v };
    { let _v = j; j -= 1; _v };
}
}
