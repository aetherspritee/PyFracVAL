! This function calculate random values of PP radii
! To this end, a lognormal PPSD is used
! rp_gstd : geometric standard deviation
! rp_g : geometric mean radii
! N: number of PP

module a_Random_PP
implicit none
contains

function lognormal_pp_radii(rp_gstd,rp_g,N)
use random
integer :: N, ii
real rp_gstd, rp_g, minVal1, maxVal1
real, allocatable :: lognormal_pp_radii(:)
allocate(lognormal_pp_radii(N))

if (rp_gstd .EQ. 1.) then
ii = 1
do while (ii .LE. N)
    lognormal_pp_radii(ii) = rp_g
        ii = ii+1
end do

else
! https://scholar.google.cl/scholar?q=Evans+Statistical+Distributions.&hl=es&as_sdt=0&as_vis=1&oi=scholart&sa=X&ved=0ahUKEwiBvtrSo93XAhXCfpAKHY9iA5IQgQMIJTAA
! Tails of the distribution are withdrawn
! 2 sigmas -> 95.5% of the distribution

minVal1 = rp_g/(rp_gstd**2.)
maxVal1 = rp_g*(rp_gstd**2.)

ii = 1
do while (ii .LE. N)
    lognormal_pp_radii(ii) = rp_g*exp(log(rp_gstd)*(random_normal()))

    if ((lognormal_pp_radii(ii) .GE. minVal1) .AND. (lognormal_pp_radii(ii) .LE. maxVal1)) then
        ii = ii+1
    end if
end do

end if
end function lognormal_pp_radii

end module a_Random_PP
