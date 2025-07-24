! THIS FUNCTION MAKES A RANDOM SAMPLE WITHOUT REPLACEMENT
! randsample(A,B): IMPUT VECTOR A, SIZE(A) = B

module RAND_SAMPLE
!use mt19937
implicit none
contains

function randsample(A,B)
real T_rand, rand_n
integer :: I, J
integer, intent(in) :: B
real A(B)
real randsample(B)

do J= 1,B
call random_number(rand_n)
I=J+INT((REAL(B-J+1))*rand_n)
T_rand = A(J)
A(J)=A(I)
A(I)=T_rand
enddo

randsample = A

end function randsample

end module RAND_SAMPLE
