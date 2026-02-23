void reverse_array(int *arr, int n) { int i=0, j=n-1; while(i<j) { int t=arr[i]; arr[i]=arr[j]; arr[j]=t; i++; j--; } }
