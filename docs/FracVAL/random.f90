MODULE random
REAL, PRIVATE             :: zero = 0.0, half = 0.5, one = 1.0, two = 2.0
PRIVATE                   :: integral
CONTAINS
REAL FUNCTION random_normal()
IMPLICIT NONE
REAL     :: s = 0.449871, t = -0.386595, a = 0.19600, b = 0.25472,           &
            r1 = 0.27597, r2 = 0.27846, u, v, x, y, q
DO
  CALL RANDOM_NUMBER(u)
  CALL RANDOM_NUMBER(v)
  v = 1.7156 * (v - half)
  x = u - s
  y = ABS(v) - t
  q = x**2 + y*(a*y - b*x)

  IF (q < r1) EXIT
  IF (q > r2) CYCLE
  IF (v**2 < -4.0*LOG(u)*u**2) EXIT
END DO

random_normal = v/u
RETURN

END FUNCTION random_normal

END module random