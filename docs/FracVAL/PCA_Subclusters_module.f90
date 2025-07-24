! THIS IS THE MAIN MODULE OF PC aggregation
module PCA_Subclusters_module
implicit none
contains

subroutine PCA_Subclusters(Number_clusters,not_able_pca,Data,i_orden,R,N,Df,kf,tol_ov,N_subcl)
use PCA_cca
implicit none
integer, intent(in) :: N, N_subcl
real, intent(in) :: Df, kf, tol_ov
real, intent(in) :: R(N)
real, intent(out) :: Data(N,4)
integer, intent(out) :: Number_clusters
logical, intent(out) :: not_able_pca
integer, allocatable, intent(out) :: i_orden(:,:)
integer, allocatable :: N_subcl_m(:)
real, allocatable :: Radius(:), Mass(:), Data_new(:,:)
real, parameter :: pi=4.0*atan(1.0)
integer Na, Number, i, j, ii, acum

Number_clusters = floor(real(N)/real(N_subcl))

if (int(mod(real(N),real(N_subcl))) .NE. 0) then
    Number_clusters = Number_clusters +1
    allocate(N_subcl_m(Number_clusters))
    N_subcl_m(1:(Number_clusters-1)) = N_subcl
    N_subcl_m(Number_clusters) = N - N_subcl*(Number_clusters-1)
else
    allocate(N_subcl_m(Number_clusters))
    N_subcl_m(1:Number_clusters) = N_subcl
end if

!parameter of each cluster (Mass,Radius,Number)
Na = 1
acum = 0

allocate(i_orden(Number_clusters,3))

do i = 1,Number_clusters
    Number = N_subcl_m(i)

    allocate(Radius(Number),Mass(Number))
    Radius = R(Na:(Na+Number-1))

    do j=1,size(Radius)
        Mass(j) = (4./3.)*pi*R(j)**3.
    end do

allocate(Data_new(Number,4))
call PCA(Data_new,not_able_pca,Number,Mass,Radius,Df,kf,tol_ov)

    if (not_able_pca) then
       return
    end if

    if (i .EQ. 1) then
        acum = Number
        do ii = 1,Number
            Data(ii,:) = Data_new(ii,:)
        end do
        i_orden(1,1:2) = (/1, acum/)
        i_orden(1,3) = acum
    else
        do ii = (acum+1),(acum+Number)
            Data(ii,:) = Data_new(ii-acum,:)
        end do
        i_orden(i,1:2) = (/(acum+1), (acum+Number)/)
        i_orden(i,3) = Number
        acum = acum + Number
    end if

    Na = Na + Number
    deallocate(Radius, Mass, Data_new)
end do

end subroutine PCA_Subclusters

end module PCA_Subclusters_module
