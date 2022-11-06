\* Source Pyomo model name=chemo *\

min 
x57:
+0 ONE_VAR_CONSTANT

s.t.

c_u_x58_:
+1 x10
+1 x13
+1 x16
+1 x19
+1 x23
+1 x26
+1 x30
+1 x33
+1 x36
+1 x39
+1 x4
+1 x42
+1 x46
+1 x49
+1 x55
+1 x7
<= 3

c_e_x59_:
+1 x1
= 59

c_e_x60_:
+1 x2
= 0

c_e_x61_:
+1 x21
= 1.0369999999999999

c_e_x62_:
+1 x28
= 0.65400000000000003

c_e_x63_:
+1 x44
= 26

c_e_x64_:
+1 x51
= 0

c_e_x65_:
+1 x52
= 1

c_e_x66_:
+1 x53
= 1

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= x1 <= +inf
    -inf <= x2 <= +inf
   0 <= x4 <= 1
   0 <= x7 <= 1
   0 <= x10 <= 1
   0 <= x13 <= 1
   0 <= x16 <= 1
   0 <= x19 <= 1
    -inf <= x21 <= +inf
   0 <= x23 <= 1
   0 <= x26 <= 1
    -inf <= x28 <= +inf
   0 <= x30 <= 1
   0 <= x33 <= 1
   0 <= x36 <= 1
   0 <= x39 <= 1
   0 <= x42 <= 1
    -inf <= x44 <= +inf
   0 <= x46 <= 1
   0 <= x49 <= 1
    -inf <= x51 <= +inf
    -inf <= x52 <= +inf
    -inf <= x53 <= +inf
   0 <= x55 <= 1
binary
  x4
  x7
  x10
  x13
  x16
  x19
  x23
  x26
  x30
  x33
  x36
  x39
  x42
  x46
  x49
  x55
end
