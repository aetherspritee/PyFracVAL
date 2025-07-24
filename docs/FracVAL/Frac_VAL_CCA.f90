! FracVAL: FracVAL: An Improved Tunable Algorithm of Cluster-Cluster Aggregation for Generation of Fractal Structures Formed by Polydisperse Primary Particles
!
! This is the cluster-cluster (CC) aggregation algorithm
! Developed by: J. Moran, A. Fuentes, F. Liu and J. Yon
! Universidad Tecnica Federico Santa Maria, Chile.
! Feel free to use this code. I just ask yoy to cite the article where we published this work.
! Any feedback on this code is welcome (jose.moranc@alumnos.usm.cl).
!

program Frac_VAL_CCA
use Ctes                !contains all the parameters
use a_Random_PP         !random primary particles generation
use RAND_SAMPLE         !random sample without replacement from a vector
use CCA_module          !Cluster-Cluster aggregation algorithm
implicit none
logical :: not_able_cca, not_able_pca

not_able_cca = .FALSE.
not_able_pca = .FALSE.
iter = 1

do while (iter <= Quantity_aggregates)

R = lognormal_pp_radii(rp_gstd,rp_g,N)
R = randsample(R,N)

print*, 'Aggregate ', iter, '/', Quantity_aggregates

call CCA_sub(not_able_cca,not_able_pca)

if (not_able_pca) then
    !The PC aggregation algorithm is not able to generate the initial clusters
    print*, 'Restarting aggregation process (PC not able to continue)'
elseif (not_able_cca) then
    !The CC aggregation algorithm is not able continue
    print*, 'Restarting aggregation process (CC not able to continue)'
else
    iter = iter + 1
end if

end do

print*,'Finshed Succesfully'
end program Frac_VAL_CCA

