fn bubble_sort(arr: *mut i32, n: i32) {
    let mut i: i32 = 0;
while (i < (n - 1)) { let mut j: i32 = 0;
while (j < ((n - i) - 1)) { if (arr[j as usize] > arr[(j + 1) as usize]) {
    let mut t: i32 = arr[j as usize];
    arr[j as usize] = arr[(j + 1) as usize];
    arr[(j + 1) as usize] = t;
}
    { let _v = j; j += 1; _v };
}
    { let _v = i; i += 1; _v };
}
}
