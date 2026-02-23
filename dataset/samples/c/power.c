long power(long base, int exp) { long r = 1; while (exp > 0) { r *= base; exp--; } return r; }
