module Ctes
    implicit none
    integer, parameter:: N=100                           !Number of PP
    real, parameter:: Df=1.79                            !Fractal dimension
    real, parameter:: kf=1.40                            !Fractal prefactor
    real, parameter:: rp_g=15.0                          !Geometric mean PP
    real, parameter:: rp_gstd=1.00                       !Geometric PP standard deviation
    integer, parameter:: Quantity_aggregates = 1        !Quantity of aggregates to be generated
    integer, parameter:: Ext_case = 0                    !Activate extreme cases
    real, parameter:: Nsubcl_perc = 0.1                  !controls the subclusters size (keep always in 10%, only increase when PC is not able to work)
    real, parameter:: tol_ov=10.**(-6.)                  !Tolerance to overlapping
    real, parameter :: pi=4.0*atan(1.0)
    real R(N)                                            !Primary particles radii and mass
    real X(N), Y(N), Z(N)                                !Coordinates of primary particles
    integer:: iter, i, j, k
end module Ctes
