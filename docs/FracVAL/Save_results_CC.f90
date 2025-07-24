module Save_results_CC
implicit none
contains

subroutine Save_results(X,Y,Z,R,N,iter)
integer, intent(in) :: N, iter
real, intent(in) :: X(N), Y(N), Z(N), R(N)
integer i
character*8 group, group_n

if ( iter.LE.9 ) then
   write(group,8) iter
else if ( iter.LE.99 ) then
   write(group,7) iter
else if ( iter.LE.999 ) then
   write(group,6) iter
else if ( iter.LE.9999 ) then
   write(group,5) iter
else if ( iter.LE.99999 ) then
   write(group,4) iter
else if ( iter.LE.999999 ) then
   write(group,3) iter
else if ( iter.LE.9999999 ) then
   write(group,2) iter
else
   write(group,1) iter
endif

if ( N.LE.9 ) then
   write(group_n,8) N
else if ( N.LE.99 ) then
   write(group_n,7) N
else if ( N.LE.999 ) then
   write(group_n,6) N
else if ( N.LE.9999 ) then
   write(group_n,5) N
else if ( N.LE.99999 ) then
   write(group_n,4) N
else if ( N.LE.999999 ) then
   write(group_n,3) N
else if ( N.LE.9999999 ) then
   write(group_n,2) N
else
   write(group_n,1) N
endif

8    format('0000000',i1)
7    format('000000',i2)
6    format('00000', i3)
5    format('0000', i4)
4    format('000', i5)
3    format('00', i6)
2    format('0', i7)
1    format(i7)

OPEN(15, file='RESULTS/'//'N_'//group_n//'_Agg_'//group//'.dat')

do i=1,N
    write(15,*) X(i), Y(i), Z(i), R(i)
end do
close(15)

end subroutine Save_results

end module Save_results_CC

