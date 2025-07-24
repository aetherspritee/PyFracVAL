! THIS IS THE MAIN MODULE OF CC aggregation
module CCA_module
implicit none
contains

subroutine CCA_sub(not_able_cca,not_able_pca)
use Ctes
use PCA_Subclusters_module
use Save_results_CC
implicit none
logical, intent(inout) :: not_able_cca, not_able_pca
integer, allocatable:: i_orden(:,:), ID_monomers(:), considered(:)
integer, allocatable :: ID_agglomerated(:,:)
real, allocatable :: X_next(:), Y_next(:), Z_next(:), R_next(:)
real, allocatable :: Xn(:), Yn(:), Zn(:), Rn(:)
integer I_t, Number_clusters, iteration, acum, N_subcl
integer iii, jjj, fill_xnew, count_k, count_other, ii,u, other, number_pairs
logical ISEMPTY
real Data(N,4)

not_able_cca = .false.
not_able_pca = .false.

if (N .LT. 50) then
N_subcl= 5              !Number of PP in each sub-cluster
else if (N .GT. 500) then
N_subcl= 50              !Number of PP in each sub-cluster
else
N_subcl= int(Nsubcl_perc*N)              !Number of PP in each sub-cluster
end if

call PCA_Subclusters(Number_clusters,not_able_pca,Data,i_orden, R,N,Df,kf,tol_ov,N_subcl)

if (not_able_pca) then
    return
end if

I_t = Number_clusters   !total number of sub-clusters

X = Data(:,1)
Y = Data(:,2)
Z = Data(:,3)
R = Data(:,4)

iteration = 1

do while (I_t .GT. 1)

i_orden = sortrows(i_orden,size(i_orden(:,1)),size(i_orden(1,:)),3)

! Generate pairs (that can be agglomerated)
allocate(ID_agglomerated(I_t,I_t))
call CCA_Generate_pairs(ID_agglomerated,not_able_CCA, X,Y,Z,R,N,I_t,i_orden,Df,kf)

    if (not_able_cca) then
        return
    end if

allocate(ID_monomers(sum(i_orden(:,3))))
call CCA_Identify_monomers(ID_monomers,i_orden)
    !CCA_Identify_monomers(ID_monomers,i_orden)
deallocate(i_orden)

! Agglomerate pairs
    allocate(considered(I_t),X_next(N),Y_next(N),Z_next(N),R_next(N))

    if (int(mod(real(I_t),2.)) .EQ. 0) then
        number_pairs = int(I_t/2)
    else
        number_pairs = floor(real(I_t)/2.)+1
    end if
    allocate(i_orden(number_pairs,3))

    k=1
    u=1
    acum = 1
    considered(1:I_t) = 0

    do while (k .LE. I_t)
        do ii = 1, size(ID_agglomerated(k,:))
            if (ID_agglomerated(k,ii) .eq. 1) then
                other = ii
                EXIT
             end if
        end do

        ISEMPTY = .TRUE.
        do ii = 1, sum(considered)
            if ((ii .EQ. other) .AND. (considered(ii) .EQ. 1)) then
                ISEMPTY = .FALSE.
                EXIT
             end if
        end do

       if ((k .NE. other) .AND. (ISEMPTY)) then
           call CCA(X,Y,Z,R,Xn,Yn,Zn,Rn,not_able_CCA, k,other,ID_monomers,Df,kf,tol_ov,N,Ext_case)
               !CCA(X,Y,Z,R,Xn,Yn,Zn,Rn,not_able_CCA, k,other,ID_monomers,Df,kf,tol_ov,N,Ext_case)

           considered(k) = 1
           considered(other) = 1

           if (not_able_CCA) then
               return
           end if

           if (sum(considered) .EQ. 2) then
               do iii = 1,size(Xn)
                    X_next(iii) = Xn(iii)
                    Y_next(iii) = Yn(iii)
                    Z_next(iii) = Zn(iii)
                    R_next(iii) = Rn(iii)
               end do
               fill_xnew = size(Xn)
           else
                do iii = (fill_xnew+1),(fill_xnew+size(Xn))
                    X_next(iii) = Xn(iii-fill_xnew)
                    Y_next(iii) = Yn(iii-fill_xnew)
                    Z_next(iii) = Zn(iii-fill_xnew)
                    R_next(iii) = Rn(iii-fill_xnew)
                end do
                fill_xnew = fill_xnew + size(Xn)
           end if

            count_k = 0
            do jjj = 1,size(ID_monomers)
                if (ID_monomers(jjj) .EQ. k) then
                count_k = count_k + 1
                end if
            end do

            count_other = 0
            do jjj = 1,size(ID_monomers)
                if (ID_monomers(jjj) .EQ. other) then
                count_other = count_other + 1
                end if
            end do

           i_orden(u,1:3) = (/acum, (acum+(count_k + count_other)-1), (count_k + count_other)/)

           acum = acum + (count_k + count_other)
           u=u+1
       end if
       k = k+1
    end do

    if (sum(considered) .LT. I_t) then
        considered(other) = 1

        count_other = 0
        do jjj = 1,size(ID_monomers)
           if (ID_monomers(jjj) .EQ. other) then
                count_other = count_other + 1
           end if
        end do

        i_orden(u,1:3) = (/acum, (acum+count_other-1), count_other/)

        ! To allocate (Xn,Yn,Zn,Rn)
        deallocate(Xn,Yn,Zn,Rn)
        allocate(Xn(count_other),Yn(count_other),Zn(count_other),Rn(count_other))

        count_other = 0
        do jjj = 1,size(ID_monomers)
           if (ID_monomers(jjj) .EQ. other) then
                count_other = count_other + 1
                Xn(count_other) = X(jjj)
                Yn(count_other) = Y(jjj)
                Zn(count_other) = Z(jjj)
                Rn(count_other) = R(jjj)
           end if
        end do

        do iii = (fill_xnew+1),(fill_xnew+size(Xn))
            X_next(iii) = Xn(iii-fill_xnew)
            Y_next(iii) = Yn(iii-fill_xnew)
            Z_next(iii) = Zn(iii-fill_xnew)
            R_next(iii) = Rn(iii-fill_xnew)
        end do
    end if

    if (int(mod(real(I_t),2.)) .EQ. 0) then
        I_t = I_t/2
    else
        I_t = floor(real(I_t)/2.)+1
    end if

    X = X_next
    Y = Y_next
    Z = Z_next
    R = R_next

    iteration = iteration + 1
    deallocate(considered,X_next,Y_next,Z_next,R_next)
    deallocate(ID_monomers,ID_agglomerated)
end do

!check for real numbers
do k=1,N
    if (isnan(X(k)) .or. isnan(Y(k))) then
        not_able_cca = .true.
        return
    elseif (isnan(Z(k)) .or. isnan(R(k))) then
        not_able_cca = .true.
        return
    end if
end do

! Save results
if (not_able_cca .OR. not_able_pca) then
    return
else
    call Save_results(X,Y,Z,R,N,iter)
end if

end subroutine CCA_sub

function sortrows(A_mat,rows,columns,c_sort)
implicit none
integer, intent(in) :: rows,columns,c_sort
integer A_mat(rows,columns)
integer, allocatable :: Am(:,:)
integer sortrows(rows,columns)
integer :: irow, krow

!nsize = size(A_mat(:,1))
allocate(Am(1,columns))

do irow = 1, rows
krow = minloc(A_mat(irow:rows, c_sort ), dim=1 ) + irow - 1

Am(1,:)     = A_mat(irow,:)
A_mat(irow,:) = A_mat(krow,:)
A_mat(krow,:) = Am(1,:)
end do

sortrows = A_mat
end function sortrows

subroutine CCA_Generate_pairs(ID_agglomerated,not_able_CCA, X,Y,Z,R,N,I_t,i_orden,Df,kf)
implicit none
integer, intent(in) :: N
real, intent(in) :: Df, kf
real, intent(in) :: X(N),Y(N),Z(N),R(N)
integer, intent(in) :: I_t
integer, intent(in) :: i_orden(I_t,3)
logical, intent(out) :: not_able_CCA
integer, intent(out) :: ID_agglomerated(I_t,I_t)
integer i, j, listo, loc, ii, size_vectors, jj, jjj
real, allocatable :: X1(:),Y1(:),Z1(:),R1(:), X2(:),Y2(:),Z2(:),R2(:)
real m1,m2,m3,rg1,rg2,rg3,R_max1,R_max2,X_cm1,Y_cm1,Z_cm1,X_cm2,Y_cm2,Z_cm2
logical :: Gamma_real
real Gamma_pc

not_able_CCA = .false.
ID_agglomerated(1:I_t,1:I_t) = 0

do i = 1,(I_t-1)
    size_vectors = i_orden(i,2)-i_orden(i,1)+1
    allocate(X1(size_vectors),Y1(size_vectors),Z1(size_vectors),R1(size_vectors))

    jjj = 0
    do jj = i_orden(i,1),i_orden(i,2)
        jjj = jjj + 1
        X1(jjj) = X(jj)
        Y1(jjj) = Y(jj)
        Z1(jjj) = Z(jj)
        R1(jjj) = R(jj)
    end do

    call CCA_AGG_properties2(m1,rg1,X_cm1,Y_cm1,Z_cm1,R_max1, X1,Y1,Z1,R1,Df,kf,jjj)
        !CCA_AGG_properties2(m, rg, X_cm, Y_cm, Z_cm, R_max,  X, Y, Z, R, Df,kf,npp)
    listo = 0
    j=1

    do while (listo .EQ. 0)
        if (j .GT. size(i_orden(:,1))) then
           not_able_CCA = .true.
           return
        end if

        if ((sum(ID_agglomerated(i,:)) .LT. 1) .AND. (sum(ID_agglomerated(:,i)) .LT. 1)) then
            if ((i .NE. j) .AND. (sum(ID_agglomerated(:,j)) .LT. 1)) then

            size_vectors = i_orden(j,2)-i_orden(j,1)+1
            allocate(X2(size_vectors),Y2(size_vectors),Z2(size_vectors),R2(size_vectors))

                jjj = 0
                do jj = i_orden(j,1),i_orden(j,2)
                    jjj = jjj + 1
                    X2(jjj) = X(jj)
                    Y2(jjj) = Y(jj)
                    Z2(jjj) = Z(jj)
                    R2(jjj) = R(jj)
                end do

                call CCA_AGG_properties2(m2,rg2,X_cm2,Y_cm2,Z_cm2,R_max2, X2,Y2,Z2,R2,Df,kf,jjj)
                    !CCA_AGG_properties2(m ,rg, X_cm, Y_cm, Z_cm, R_max,  X, Y, Z, R, Df,kf,npp)
                m3 = m1+m2
                rg3 = (exp(sum(log([R1, R2]))/real(size(log([R1, R2])))))*(real((size(R1)+size(R2)))/kf)**(1./Df)

                if((m3**2.)*(rg3**2.) .GT. (m3*(m1*rg1**2.+m2*rg2**2.))) then
                    Gamma_pc = sqrt(((m3**2.)*(rg3**2.)-m3*(m1*rg1**2.+m2*rg2**2.))/(m1*m2))
                    Gamma_real = .true.
                else
                    Gamma_real = .false.
                end if

                if ((Gamma_pc .LT. (R_max1+R_max2)) .AND. (Gamma_real)) then
                    ID_agglomerated(i,j) = 1
                    ID_agglomerated(j,i) = 1
                    listo = 1
                end if
                deallocate(X2,Y2,Z2,R2)
            end if
        else
            listo = 1
        end if
        j = j+1
    end do
deallocate(X1,Y1,Z1,R1)
end do

if (int(mod(real(I_t),2.)) .NE. 0) then
    do ii = 1, size(ID_agglomerated(1,:))
        if (sum(ID_agglomerated(:,ii)) .eq. 0) then
            loc = ii
            exit
        endif
    end do
    ID_agglomerated(loc,loc) = 1
end if
end subroutine CCA_Generate_pairs

subroutine CCA_AGG_properties2(m,rg,X_cm,Y_cm,Z_cm,R_max, X,Y,Z,R,Df,kf,npp)
implicit none
integer, intent(in) :: npp
real, intent(out) :: m, rg, X_cm,Y_cm,Z_cm, R_max
real, intent(in) :: X(npp),Y(npp),Z(npp),R(npp)
real, intent(in):: Df, kf
real m_vec(npp), Ri(npp)
real sum_xm,sum_ym,sum_zm
integer i
real, parameter :: pi=4.0*atan(1.0)

sum_xm = 0.
sum_ym = 0.
sum_zm = 0.

do i=1,npp
m_vec(i) = (4.*pi/3.)*(R(i))**3.

sum_xm = sum_xm + X(i)*m_vec(i)
sum_ym = sum_ym + Y(i)*m_vec(i)
sum_zm = sum_zm + Z(i)*m_vec(i)
end do

m = sum(m_vec)

X_cm = sum_xm/m
Y_cm = sum_ym/m
Z_cm = sum_zm/m

rg = (exp(sum(log(R))/real(size(log(R)))))*(real(npp)/kf)**(1./Df)

Ri(1:npp) = 0.
do i = 1,npp
    Ri(i) = sqrt((X_cm-X(i))**2. + (Y_cm-Y(i))**2. + (Z_cm-Z(i))**2.)
end do

R_max = maxval(Ri)
end subroutine CCA_AGG_properties2

subroutine CCA_Identify_monomers(ID_monomers,i_orden)
implicit none
integer,allocatable, intent(in) :: i_orden(:,:)
integer, allocatable, intent(out):: ID_monomers(:)
integer i,j

allocate(ID_monomers(sum(i_orden(:,3))))
ID_monomers(1:sum(i_orden(:,3))) = 0

do i = 1,size(i_orden(:,1))
    do j = i_orden(i,1),i_orden(i,2)
        ID_monomers(j) = i
    end do
end do
end subroutine CCA_Identify_monomers

subroutine CCA(X,Y,Z,R,Xn,Yn,Zn,Rn,not_able_CCA, k,other,ID_monomers,Df,kf,tol_ov,N,Ext_case)
implicit none
integer, intent(in) :: k, other, N, Ext_case
real, intent(in) :: Df,kf,tol_ov
integer, intent(in) :: ID_monomers(N)
real, intent(inout) :: X(N),Y(N),Z(N),R(N)
logical, intent(out):: not_able_CCA
real, allocatable, intent(out) :: Xn(:),Yn(:),Zn(:),Rn(:)

real, parameter :: pi=4.0*atan(1.0)
integer ii, monomers_1, monomers_2
integer n1, n2, n3, lista_suma, prev_cand1, prev_cand2, intento
real, allocatable :: X1(:), Y1(:), Z1(:), R1(:)
real, allocatable :: X2(:), Y2(:), Z2(:), R2(:)
integer, allocatable :: list(:,:)
real rg1, rg2, rg3, m1, m2, m3
real X_cm1,Y_cm1,Z_cm1,R_max1
real X_cm2,Y_cm2,Z_cm2,R_max2, Cov_max
real CM1(3), CM2(3)
real, allocatable :: COR1(:,:), COR2(:,:)
logical Gamma_real
real i_vec(3), j_vec(3), vec_0(4)
real Gamma_pc,theta_a, x0,y0,z0,r0

not_able_CCA = .false.

monomers_1 = 0
do ii = 1,N
    if (ID_monomers(ii) .EQ. k) then
        monomers_1 = monomers_1 + 1
    end if
end do

allocate(X1(monomers_1),Y1(monomers_1),Z1(monomers_1),R1(monomers_1))

monomers_1 = 0
do ii = 1,N
    if (ID_monomers(ii) .EQ. k) then
        monomers_1 = monomers_1 + 1
        X1(monomers_1) = X(ii)
        Y1(monomers_1) = Y(ii)
        Z1(monomers_1) = Z(ii)
        R1(monomers_1) = R(ii)
    end if
end do

n1 = size(X1)
call CCA_AGG_properties2(m1,rg1,X_cm1,Y_cm1,Z_cm1,R_max1, X1,Y1,Z1,R1,Df,kf,n1)
    !CCA_AGG_properties2(m, rg, X_cm, Y_cm, Z_cm, R_max,  X, Y, Z,R,  Df,kf,npp)

monomers_2 = 0
do ii = 1,N
    if (ID_monomers(ii) .EQ. other) then
        monomers_2 = monomers_2 + 1
    end if
end do

allocate(X2(monomers_2),Y2(monomers_2),Z2(monomers_2),R2(monomers_2))

monomers_2 = 0
do ii = 1,N
    if (ID_monomers(ii) .EQ. other) then
        monomers_2 = monomers_2 + 1
        X2(monomers_2) = X(ii)
        Y2(monomers_2) = Y(ii)
        Z2(monomers_2) = Z(ii)
        R2(monomers_2) = R(ii)
    end if
end do

n2 = size(X2)
call CCA_AGG_properties2(m2,rg2,X_cm2,Y_cm2,Z_cm2,R_max2, X2,Y2,Z2,R2,Df,kf,n2)
    !CCA_AGG_properties2(m, rg, X_cm, Y_cm, Z_cm, R_max,  X, Y, Z,R,  Df,kf,npp)

m3 = m1 + m2
n3 = n1 + n2

rg3 = (exp(sum(log([R1, R2]))/real(size(log([R1, R2])))))*(real(n3)/kf)**(1./Df)

if((m3**2.)*(rg3**2.) .GT. (m3*(m1*rg1**2.+m2*rg2**2.))) then
     Gamma_pc = sqrt(((m3**2.)*(rg3**2.)-m3*(m1*rg1**2.+m2*rg2**2.))/(m1*m2))
     Gamma_real = .true.
else
     Gamma_real = .false.
end if

CM1(1) = X_cm1
CM1(2) = Y_cm1
CM1(3) = Z_cm1

CM2(1) = X_cm2
CM2(2) = Y_cm2
CM2(3) = Z_cm2

allocate(list(n1,n2))
call CCA_Random_select_list(list,n1,X1,Y1,Z1,R1,CM1, n2,X2,Y2,Z2,R2,CM2,Gamma_pc,Gamma_real,Ext_case)
    !CCA_Random_select_list(list,n1,X1,Y1,Z1,R1,CM1, n2,X2,Y2,Z2,R2,CM2,Gamma_pc,Gamma_real,Ext_case)

allocate(COR1(n1,4),COR2(n2,4))

lista_suma = 0
do while (lista_suma .EQ. 0)
    prev_cand1 = 0
    Cov_max = 1.

    do while (Cov_max .GT. tol_ov)
        if (sum(list) .GT. 1) then
            call CCA_Random_select_list_pick_one_1(list,prev_cand1)
            !Step 4: Select a candidate
            prev_cand2 = 0
            call CCA_Random_select_list_pick_one_2(list,prev_cand1,prev_cand2)
        else
            not_able_CCA = .true.
            return
        end if

        intento = 1
        ! Step 5: sticking process

        CM1(1) = X_cm1
        CM1(2) = Y_cm1
        CM1(3) = Z_cm1

        CM2(1) = X_cm2
        CM2(2) = Y_cm2
        CM2(3) = Z_cm2

        COR1(:,1) = X1
        COR1(:,2) = Y1
        COR1(:,3) = Z1
        COR1(:,4) = R1

        COR2(:,1) = X2
        COR2(:,2) = Y2
        COR2(:,3) = Z2
        COR2(:,4) = R2

call CCA_Sticking_process_v1(COR1,CM1,n1,COR2,CM2,n2,theta_a,vec_0,i_vec,j_vec,prev_cand1,prev_cand2,Gamma_pc,Gamma_real,Ext_case)
    !CCA_Sticking_process_v1(COR1,CM1,n1,COR2,CM2,n2,theta_a,vec_0,i_vec,j_vec,prev_cand1,prev_cand2,Gamma_pc,Gamma_real,Ext_case)

        X1(:) = COR1(:,1)
        Y1(:) = COR1(:,2)
        Z1(:) = COR1(:,3)

        X2(:) = COR2(:,1)
        Y2(:) = COR2(:,2)
        Z2(:) = COR2(:,3)

        X_cm2 = CM2(1)
        Y_cm2 = CM2(2)
        Z_cm2 = CM2(3)

        x0 = vec_0(1)
        y0 = vec_0(2)
        z0 = vec_0(3)
        r0 = vec_0(4)

        ! Overlaping check
         call CCA_Overlapping_check(Cov_max, X1,Y1,Z1,R1,n1, X2,Y2,Z2,R2,n2)

        do while ((Cov_max .GT. tol_ov) .AND. (intento .LT. 360))
            ! Step 6: Rotation
            vec_0(1) = x0
            vec_0(2) = y0
            vec_0(3) = z0
            vec_0(4) = r0

            CM2(1) = X_cm2
            CM2(2) = Y_cm2
            CM2(3) = Z_cm2

            CALL CCA_Sticking_process_v1_reintento(theta_a,vec_0,i_vec,j_vec,X2,Y2,Z2,n2,prev_cand2,CM2)
                !CCA_Sticking_process_v1_reintento(theta_a,vec_0,i_vec,j_vec,X2,Y2,Z2,n2,prev_cand2,CM2)

            ! Overlaping check
            call CCA_Overlapping_check(Cov_max, X1,Y1,Z1,R1,n1, X2,Y2,Z2,R2,n2)
            intento = intento + 1

                if ((INT(mod(real(intento),real((359)))) .EQ. 0) .AND. (sum(list(prev_cand1,:)) .GT. 1)) then
                ! Step 4: Select a new candidate
                call CCA_Random_select_list_pick_one_2(list,prev_cand1,prev_cand2)

                CM1(1) = X_cm1
                CM1(2) = Y_cm1
                CM1(3) = Z_cm1

                CM2(1) = X_cm2
                CM2(2) = Y_cm2
                CM2(3) = Z_cm2

                COR1(:,1) = X1
                COR1(:,2) = Y1
                COR1(:,3) = Z1
                COR1(:,4) = R1

                COR2(:,1) = X2
                COR2(:,2) = Y2
                COR2(:,3) = Z2
                COR2(:,4) = R2

call CCA_Sticking_process_v1(COR1,CM1,n1,COR2,CM2,n2,theta_a,vec_0,i_vec,j_vec,prev_cand1,prev_cand2,Gamma_pc,Gamma_real,Ext_case)
    !CCA_Sticking_process_v1(COR1,CM1,n1,COR2,CM2,n2,theta_a,vec_0,i_vec,j_vec,prev_cand1,prev_cand2,Gamma_pc,Gamma_real,Ext_case)
                X1(:) = COR1(:,1)
                Y1(:) = COR1(:,2)
                Z1(:) = COR1(:,3)

                X2(:) = COR2(:,1)
                Y2(:) = COR2(:,2)
                Z2(:) = COR2(:,3)

                X_cm2 = CM2(1)
                Y_cm2 = CM2(2)
                Z_cm2 = CM2(3)

                x0 = vec_0(1)
                y0 = vec_0(2)
                z0 = vec_0(3)
                r0 = vec_0(4)

                intento = 1
                call CCA_Overlapping_check(Cov_max, X1,Y1,Z1,R1,n1, X2,Y2,Z2,R2,n2)
                end if
         end do
        lista_suma = sum(list(prev_cand1,:))
    end do
end do

monomers_1 = 0
do ii=1, N
    if (ID_monomers(ii) .EQ. k) then
        monomers_1 = monomers_1 + 1
        X(ii) = X1(monomers_1)
        Y(ii) = Y1(monomers_1)
        Z(ii) = Z1(monomers_1)
        R(ii) = R1(monomers_1)
    end if
end do

monomers_2 = 0
do ii=1, N
    if (ID_monomers(ii) .EQ. other) then
        monomers_2 = monomers_2 + 1
        X(ii) = X2(monomers_2)
        Y(ii) = Y2(monomers_2)
        Z(ii) = Z2(monomers_2)
        R(ii) = R2(monomers_2)
    end if
end do

allocate(Xn(n1+n2),Yn(n1+n2),Zn(n1+n2),Rn(n1+n2))
do ii = 1,n1
Xn(ii) = X1(ii)
Yn(ii) = Y1(ii)
Zn(ii) = Z1(ii)
Rn(ii) = R1(ii)
end do

do ii = (n1+1),(n1+n2)
Xn(ii) = X2(ii-n1)
Yn(ii) = Y2(ii-n1)
Zn(ii) = Z2(ii-n1)
Rn(ii) = R2(ii-n1)
end do

if (Cov_max .GT. tol_ov) then
            not_able_CCA = .true.
            return
end if

end subroutine CCA

subroutine CCA_Random_select_list(list,n1,X1,Y1,Z1,R1,CM1,n2,X2,Y2,Z2,R2,CM2,Gamma_pc,Gamma_real,Ext_case)
implicit none
logical, intent(in):: Gamma_real
integer, intent(in) :: n1,n2, Ext_case
real, intent(in) :: Gamma_pc
real, intent(in) :: X1(n1),Y1(n1),Z1(n1),R1(n1)
real, intent(in) :: X2(n2),Y2(n2),Z2(n2),R2(n2)
real, intent(in) :: CM1(3), CM2(3)
integer, intent(out) :: list(n1,n2)
integer i,j
real Dimax, Dimin, Djmax, Djmin
real X_cm1, Y_cm1, Z_cm1
real X_cm2, Y_cm2, Z_cm2

X_cm1 = CM1(1)
Y_cm1 = CM1(2)
Z_cm1 = CM1(3)

X_cm2 = CM2(1)
Y_cm2 = CM2(2)
Z_cm2 = CM2(3)

list(1:n1,1:n2) = 0

if (Gamma_real .AND. (Ext_case .EQ. 1)) then
    do i = 1,n1
        Dimin = sqrt((X1(i)-X_cm1)**2. + (Y1(i)-Y_cm1)**2. + (Z1(i)-Z_cm1)**2.) - R1(i)
        Dimax = sqrt((X1(i)-X_cm1)**2. + (Y1(i)-Y_cm1)**2. + (Z1(i)-Z_cm1)**2.) + R1(i)

        do j = 1,n2
            Djmin = sqrt((X2(j)-X_cm2)**2. + (Y2(j)-Y_cm2)**2. + (Z2(j)-Z_cm2)**2.) - R2(j)
            Djmax = sqrt((X2(j)-X_cm2)**2. + (Y2(j)-Y_cm2)**2. + (Z2(j)-Z_cm2)**2.) + R2(j)

            if ((Dimax + Djmax) .GT. Gamma_pc) then
                if (ABS(Djmax-Dimax) .LT. Gamma_pc) then
                    list(i,j) = 1
                else if ((Djmax-Dimax .GT. Gamma_pc) .AND. (Djmin-Dimax .LT. Gamma_pc)) then
                    list(i,j) = 1
                else if ((Dimax-Djmax .GT. Gamma_pc) .AND. (Dimin-Djmax .LT. Gamma_pc)) then
                    list(i,j) = 1
                endif
            end if

        end do
    end do
else if (Gamma_real .AND. (Ext_case .EQ. 0)) then
    do i = 1,n1
        Dimax = sqrt((X1(i)-X_cm1)**2. + (Y1(i)-Y_cm1)**2. + (Z1(i)-Z_cm1)**2.) + R1(i)
        do j = 1,n2
            Djmax = sqrt((X2(j)-X_cm2)**2. + (Y2(j)-Y_cm2)**2. + (Z2(j)-Z_cm2)**2.) + R2(j)

            if (((Dimax + Djmax) .GT. Gamma_pc) .AND. (ABS(Djmax-Dimax) .LT. Gamma_pc)) then
               list(i,j) = 1
            end if

        end do
    end do
else
    return
end if
end subroutine CCA_Random_select_list

subroutine CCA_Random_select_list_pick_one_1(list,prev_cand1)
implicit none
integer, intent(inout) :: prev_cand1
integer, intent(inout) :: list(:,:)
integer i,ii,jj,selected,sel,selected_real
real rand_n
integer, allocatable :: list_suma(:), list2(:)

if (prev_cand1 .GT. 0) then
    list(prev_cand1,:) = list(prev_cand1,:)*0
end if

! random selection of a monomer (inicialmente sin diferencia)
allocate(list_suma(size(list(:,1))))
do i = 1,size(list(:,1))
    list_suma(i) = sum(list(i,:))
end do

allocate(list2(size(pack(list_suma,list_suma .GT. 0))))
list2 = pack(list_suma,list_suma .GT. 0)

call RANDOM_NUMBER(rand_n)
selected = 1+INT((REAL(size(list2)-1))*rand_n)

sel = 0
jj = 0

do ii = 1,size(list_suma)
    if (list_suma(ii) .GT. 0) then
        jj = jj+1
        sel = jj
    end if

    if  (sel .EQ. selected) then
        selected_real = ii
        EXIT
    end if
end do

prev_cand1 = selected_real
end subroutine CCA_Random_select_list_pick_one_1

subroutine CCA_Random_select_list_pick_one_2(list,prev_cand1,prev_cand2)
implicit none
integer, intent(in) :: prev_cand1
integer, intent(inout) :: prev_cand2
integer, intent(inout) :: list(:,:)
integer ii,jj,selected,sel,selected_real
real rand_n
integer, allocatable :: list_suma(:), list2(:)

if (prev_cand2 .GT. 0) then
    list(prev_cand1,prev_cand2) = list(prev_cand1,prev_cand2)*0
end if

! random selection of a monomer
allocate(list_suma(size(list(prev_cand1,:))))
list_suma = list(prev_cand1,:)

allocate(list2(size(pack(list_suma,list_suma .GT. 0))))
list2 = pack(list_suma,list_suma .GT. 0)

call RANDOM_NUMBER(rand_n)
selected = 1+INT((REAL(size(list2)-1))*rand_n)

sel = 0
jj = 0

do ii = 1,size(list_suma)
    if (list_suma(ii) .GT. 0) then
        jj = jj+1
        sel = jj
    end if

    if  (sel .EQ. selected) then
        selected_real = ii
        EXIT
    end if
end do

prev_cand2 = selected_real
end subroutine CCA_Random_select_list_pick_one_2

subroutine CCA_Sticking_process_v1(COR1,CM1,n1,COR2,CM2,n2,theta_a,vec_0,i_vec,j_vec,prev_cand1,&
                                   prev_cand2,Gamma_pc,Gamma_real,Ext_case)
!use Save_results_CC
implicit none
integer, intent(in) :: n1, n2, Ext_case
integer, intent(in) :: prev_cand1, prev_cand2
real, intent(inout) ::  COR1(n1,4), COR2(n2,4)
real, intent(in) ::  Gamma_pc
logical, intent(in) :: Gamma_real
real, intent(in) :: CM1(3)
real, intent(inout) :: CM2(3)
real, intent(out) :: vec_0(4)
real, intent(out) :: theta_a
real, intent(out) :: i_vec(3),j_vec(3)

real x0,y0,z0,r0
real sphere_1(4), sphere_2(4), spheres_1(5), spheres_2(5)
real X1(n1),Y1(n1),Z1(n1),R1(n1)
real X2(n2),Y2(n2),Z2(n2),R2(n2)
real X1_new(n1),Y1_new(n1),Z1_new(n1)
real X2_new(n2),Y2_new(n2),Z2_new(n2)
real X_cm1, Y_cm1, Z_cm1
real X_cm2, Y_cm2, Z_cm2
real vect_x, vect_y, vect_z, vect_mag
real desplazamiento_x, desplazamiento_y, desplazamiento_z
real x_cm11,y_cm11,z_cm11
real x_cm22,y_cm22,z_cm22
real x1_sph1,y1_sph1,z1_sph1,r1_sph1,D1max,D1min
real x2_sph2,y2_sph2,z2_sph2,r2_sph2,D2max, D2min
real v1(3), v2(3), s_vec(3), As(3,3), Identity(3,3), Rot(3,3), New_c(3), u_s1_cm1(3)
real angle, displacement_s1
real x,y,z
integer i, case

if (Gamma_real) then
X1(:) = COR1(:,1)
Y1(:) = COR1(:,2)
Z1(:) = COR1(:,3)
R1(:) = COR1(:,4)

X2(:) = COR2(:,1)
Y2(:) = COR2(:,2)
Z2(:) = COR2(:,3)
R2(:) = COR2(:,4)

X_cm1 = CM1(1)
Y_cm1 = CM1(2)
Z_cm1 = CM1(3)

X_cm2 = CM2(1)
Y_cm2 = CM2(2)
Z_cm2 = CM2(3)

vect_x = X1(prev_cand1) - X_cm1
vect_y = Y1(prev_cand1) - Y_cm1
vect_z = Z1(prev_cand1) - Z_cm1

vect_mag = sqrt(vect_x**2. + vect_y**2. + vect_z**2.)

vect_x = vect_x/vect_mag
vect_y = vect_y/vect_mag
vect_z = vect_z/vect_mag

! Center of mass of aggregate 2
x_cm22 = X_cm1 + Gamma_pc*vect_x
y_cm22 = Y_cm1 + Gamma_pc*vect_y
z_cm22 = Z_cm1 + Gamma_pc*vect_z

! New coordinates monomers aggregate 2
desplazamiento_x = x_cm22 - X_cm2
desplazamiento_y = y_cm22 - Y_cm2
desplazamiento_z = z_cm22 - Z_cm2

X2_new = X2 + desplazamiento_x
Y2_new = Y2 + desplazamiento_y
Z2_new = Z2 + desplazamiento_z

! Sphere 1: CM1, dist1
x1_sph1 = X_cm1
y1_sph1 = Y_cm1
z1_sph1 = Z_cm1
!r1_sph1 = sqrt((X1(prev_cand1)-X_cm1)**2. + (Y1(prev_cand1)-Y_cm1)**2. + (Z1(prev_cand1)-Z_cm1)**2.)
D1min = sqrt((X1(prev_cand1)-X_cm1)**2. +  (Y1(prev_cand1)-Y_cm1)**2. +  (Z1(prev_cand1)-Z_cm1)**2.) &
        -  R1(prev_cand1)
D1max = sqrt((X1(prev_cand1)-X_cm1)**2. +  (Y1(prev_cand1)-Y_cm1)**2. +  (Z1(prev_cand1)-Z_cm1)**2.) &
        +  R1(prev_cand1)

! Sphere 2: CM2, dist2
x2_sph2 = x_cm22
y2_sph2 = y_cm22
z2_sph2 = z_cm22
!r2_sph2 = sqrt((X2_new(prev_cand2)-x_cm22)**2. + (Y2_new(prev_cand2)-y_cm22)**2. + (Z2_new(prev_cand2)-z_cm22)**2.)
D2min = sqrt((X2_new(prev_cand2)-X_cm22)**2. +  (Y2_new(prev_cand2)-Y_cm22)**2. +  (Z2_new(prev_cand2)-Z_cm22)**2.) &
        -  R2(prev_cand2)
D2max = sqrt((X2_new(prev_cand2)-X_cm22)**2. +  (Y2_new(prev_cand2)-Y_cm22)**2. +  (Z2_new(prev_cand2)-Z_cm22)**2.) &
        +  R2(prev_cand2)

!sphere_1 = (/x1_sph1, y1_sph1, z1_sph1, r1_sph1/)
!sphere_2 = (/x2_sph2, y2_sph2, z2_sph2, r2_sph2/)
!call CCA_Two_sphere_intersection(x,y,z,theta_a,vec_0,i_vec,j_vec,sphere_1,sphere_2)
spheres_1 = (/x1_sph1, y1_sph1, z1_sph1, D1min, D1max/)
spheres_2 = (/x2_sph2, y2_sph2, z2_sph2, D2min, D2max/)

if (Ext_case .EQ. 1) then

   if (ABS(D2max-D1max) .LT. Gamma_pc) then
     case = 1
     call Random_point_SC(x,y,z,case,spheres_1, spheres_2)
     u_s1_cm1 = (/(X_cm1-x), (Y_cm1-y), (Z_cm1-z)/)

  else if ((D2max-D1max .GT. Gamma_pc) .AND. (D2min-D1max .LT. Gamma_pc)) then
     case = 2
     call Random_point_SC(x,y,z,case,spheres_1, spheres_2)
     u_s1_cm1 = (/(X_cm1-x), (Y_cm1-y), (Z_cm1-z)/)

  else if ((D1max-D2max .GT. Gamma_pc) .AND. (D1min-D2max .LT. Gamma_pc)) then
     case = 3
     call Random_point_SC(x,y,z,case,spheres_1, spheres_2)
     u_s1_cm1 = (/(x-X_cm1), (y-Y_cm1), (z-Z_cm1)/)
   end if

else if (Ext_case .EQ. 0) then
     r1_sph1 = D1max
     r2_sph2 = D2max

     sphere_1 = (/x1_sph1, y1_sph1, z1_sph1, r1_sph1/)
     sphere_2 = (/x2_sph2, y2_sph2, z2_sph2, r2_sph2/)
     call CCA_Two_sphere_intersection(x,y,z,theta_a,vec_0,i_vec,j_vec,sphere_1,sphere_2)
     u_s1_cm1 = (/(X_cm1-x), (Y_cm1-y), (Z_cm1-z)/)

end if

u_s1_cm1 = u_s1_cm1/norm2(u_s1_cm1)
displacement_s1 = R1(prev_cand1)

x = x + displacement_s1*u_s1_cm1(1)
y = y + displacement_s1*u_s1_cm1(2)
z = z + displacement_s1*u_s1_cm1(3)

! Update coordinates of PP belonging to aggregate 1
! Based on Euler-Rodriguez formula
x_cm11 = X_cm1
y_cm11 = Y_cm1
z_cm11 = Z_cm1

X1_new = X1
Y1_new = Y1
Z1_new = Z1

v1 = (/(X1_new(prev_cand1)-x_cm11), (Y1_new(prev_cand1)-y_cm11), (Z1_new(prev_cand1)-z_cm11)/)
v2 = (/(x-x_cm11), (y-y_cm11), (z-z_cm11)/)
s_vec = cross(v1,v2)/norm2(cross(v1,v2))
angle = acos(DOT_PRODUCT(v1,v2)/(norm2(v1)*norm2(v2)))

As = RESHAPE((/0., s_vec(3), -s_vec(2), -s_vec(3), 0., s_vec(1), s_vec(2), -s_vec(1), 0./),(/3,3/))
Identity = RESHAPE((/1., 0., 0., 0., 1., 0., 0., 0., 1./),(/3,3/))
Rot = Identity +sin(angle)*As + (1.-cos(angle))*MATMUL(As,As)

do i = 1,n1
    New_c = MATMUL(Rot,(/(X1_new(i)-x_cm11), (Y1_new(i)-y_cm11), (Z1_new(i)-z_cm11)/))

    X1_new(i) = x_cm11 + New_c(1)
    Y1_new(i) = y_cm11 + New_c(2)
    Z1_new(i) = z_cm11 + New_c(3)
end do

! Parameters
! (x1, y1, z1) -> center sphere 1: SELECTED sphere from aggregate 1
! (x2, y2, z2) -> center sphere 2: CM of aggregate 2 (line 40)

! Sphere 2 information
r2_sph2 = sqrt((X2_new(prev_cand2)-x_cm22)**2.+(Y2_new(prev_cand2)-y_cm22)**2.+(Z2_new(prev_cand2)-z_cm22)**2.)
x2_sph2 = x_cm22
y2_sph2 = y_cm22
z2_sph2 = z_cm22

! Two spheres interception
r1_sph1 = R1(prev_cand1) + R2(prev_cand2)
x1_sph1 = X1_new(prev_cand1)
y1_sph1 = Y1_new(prev_cand1)
z1_sph1 = Z1_new(prev_cand1)

sphere_1 = (/x1_sph1, y1_sph1, z1_sph1, r1_sph1/)
sphere_2 = (/x2_sph2, y2_sph2, z2_sph2, r2_sph2/)

call CCA_Two_sphere_intersection(x,y,z,theta_a,vec_0,i_vec,j_vec,sphere_1,sphere_2)

x0 = vec_0(1)
y0 = vec_0(2)
z0 = vec_0(3)
r0 = vec_0(4)

! Update coordinates of PP belonging to aggregate 2
! Based on Euler-Rodriguez formula
v1 = (/(X2_new(prev_cand2)-x_cm22), (Y2_new(prev_cand2)-y_cm22), (Z2_new(prev_cand2)-z_cm22)/)
v2 = (/(x-x_cm22), (y-y_cm22), (z-z_cm22)/)
s_vec = cross(v1,v2)/norm2(cross(v1,v2))

angle = acos(DOT_PRODUCT(v1,v2)/(norm2(v1)*norm2(v2)))

As = RESHAPE((/0., s_vec(3), -s_vec(2), -s_vec(3), 0., s_vec(1), s_vec(2), -s_vec(1), 0./),(/3,3/))
Identity = RESHAPE((/1., 0., 0., 0., 1., 0., 0., 0., 1./),(/3,3/))
Rot = Identity + sin(angle)*As + (1.-cos(angle))*MATMUL(As,As)

do i = 1,n2
    New_c = MATMUL(Rot, (/(X2_new(i)-x_cm22), (Y2_new(i)-y_cm22), (Z2_new(i)-z_cm22)/))

    X2_new(i) = x_cm22 + New_c(1)
    Y2_new(i) = y_cm22 + New_c(2)
    Z2_new(i) = z_cm22 + New_c(3)
end do

vec_0(1) = x0
vec_0(2) = y0
vec_0(3) = z0
vec_0(4) = r0

CM2(1) = x_cm22
CM2(2) = y_cm22
CM2(3) = z_cm22

COR1(:,1) = X1_new(:)
COR1(:,2) = Y1_new(:)
COR1(:,3) = Z1_new(:)

COR2(:,1) = X2_new(:)
COR2(:,2) = Y2_new(:)
COR2(:,3) = Z2_new(:)
else
    return
end if

end subroutine CCA_Sticking_process_v1

subroutine CCA_Two_sphere_intersection(x,y,z,theta,vec_0,i_vec,j_vec,sphere_1,sphere_2)
implicit none
real, intent(in) :: sphere_1(4), sphere_2(4)
real, intent(out) :: theta
real, intent(out) :: i_vec(3), j_vec(3), vec_0(4)
real, intent(out) :: x,y,z

real x0,y0,z0,r0
real x1_sph1,y1_sph1,z1_sph1,r1_sph1
real x2_sph2,y2_sph2,z2_sph2,r2_sph2
real k_vec(3)
real A,B,C,D, AmBdC, distance, alpha_0, u, t
real, parameter :: pi=4.0*atan(1.0)

x1_sph1 = sphere_1(1)
y1_sph1 = sphere_1(2)
z1_sph1 = sphere_1(3)
r1_sph1 = sphere_1(4)

x2_sph2 = sphere_2(1)
y2_sph2 = sphere_2(2)
z2_sph2 = sphere_2(3)
r2_sph2 = sphere_2(4)

A = 2.*(x2_sph2-x1_sph1)
B = 2.*(y2_sph2-y1_sph1)
C = 2.*(z2_sph2-z1_sph1)
D = x1_sph1**2.0 - x2_sph2**2.0 + y1_sph1**2.0 - y2_sph2**2.0 + z1_sph1**2.0 - z2_sph2**2.0 - r1_sph1**2.0 + r2_sph2**2.0

t = (x1_sph1*A + y1_sph1*B + z1_sph1*C + D)/(A*(x1_sph1-x2_sph2) + B*(y1_sph1-y2_sph2) + C*(z1_sph1-z2_sph2))

x0 = x1_sph1 + t*(x2_sph2-x1_sph1)
y0 = y1_sph1 + t*(y2_sph2-y1_sph1)
z0 = z1_sph1 + t*(z2_sph2-z1_sph1)

distance = sqrt((x2_sph2-x1_sph1)**2. + (y2_sph2-y1_sph1)**2. + (z2_sph2-z1_sph1)**2.)

alpha_0 = acos((r1_sph1**2. + distance**2. - r2_sph2**2.)/(2.*r1_sph1*distance))
r0 = r1_sph1*sin(alpha_0)

AmBdC = -(A+B)/C

k_vec = (/A, B, C/)/sqrt(A**2.+B**2.+C**2.)
i_vec = (/1., 1., AmBdC/)/sqrt(1.+1.+(AmBdC)**2.)
j_vec = cross(k_vec,i_vec)

call random_number(u)
theta = 2.*pi*u

! parametrized
x = x0 + r0*cos(theta)*i_vec(1) + r0*sin(theta)*j_vec(1)
y = y0 + r0*cos(theta)*i_vec(2) + r0*sin(theta)*j_vec(2)
z = z0 + r0*cos(theta)*i_vec(3) + r0*sin(theta)*j_vec(3)

vec_0(1) = x0
vec_0(2) = y0
vec_0(3) = z0
vec_0(4) = r0
end subroutine CCA_Two_sphere_intersection

FUNCTION cross(a, b)
  REAL, DIMENSION(3) :: cross
  REAL, DIMENSION(3), INTENT(IN) :: a, b

  cross(1) = a(2) * b(3) - a(3) * b(2)
  cross(2) = a(3) * b(1) - a(1) * b(3)
  cross(3) = a(1) * b(2) - a(2) * b(1)
END FUNCTION cross

subroutine Random_point_SC(x,y,z,case,spheres_1,spheres_2)
implicit none
real, intent(in) :: spheres_1(5), spheres_2(5)
integer, intent(in) :: case
real, intent(out) :: x,y,z
real u,v,phi_r,theta_r, angle, r1_sph1_min, r1_sph1_max, r2_sph2_min, r2_sph2_max
real v1(3), v2(3), s_vec(3), As(3,3), Identity(3,3), Rot(3,3), New_c(3),r12(3)
real sphere_1(4), sphere_2(4)
real phi_cr_min, phi_cr_max, norm12
real x1_sph1,y1_sph1,z1_sph1,r1_sph1, x2_sph2,y2_sph2,z2_sph2
real, parameter :: pi=4.0*atan(1.0)

sphere_1(1) = spheres_1(1)
sphere_1(2) = spheres_1(2)
sphere_1(3) = spheres_1(3)
r1_sph1_min = spheres_1(4)
r1_sph1_max = spheres_1(5)

sphere_2(1) = spheres_2(1)
sphere_2(2) = spheres_2(2)
sphere_2(3) = spheres_2(3)
r2_sph2_min = spheres_2(4)
r2_sph2_max = spheres_2(5)

if (case .EQ. 1) then
    sphere_1(4) = r1_sph1_max
    sphere_2(4) = r2_sph2_max
    call Spherical_cap_angle(phi_cr_max,sphere_1,sphere_2)

    norm12 = norm2((/(sphere_1(1)-sphere_2(1)), (sphere_1(2)-sphere_2(2)), (sphere_1(3)-sphere_2(3))/))
    if ((r1_sph1_max + r2_sph2_min) .GT. norm12) then
        sphere_1(4) = r1_sph1_max
        sphere_2(4) = r2_sph2_min
        call Spherical_cap_angle(phi_cr_min,sphere_1,sphere_2)
    else
        phi_cr_min = 0.
    end if
    r1_sph1 = r1_sph1_max

else if (case .EQ. 2) then
    sphere_1(4) = r1_sph1_max
    sphere_2(4) = r2_sph2_min
    call Spherical_cap_angle(phi_cr_max,sphere_1,sphere_2)
    phi_cr_min = 0.
    r1_sph1 = r1_sph1_max

else if (case .EQ. 3) then
    sphere_1(4) = r1_sph1_min
    sphere_2(4) = r2_sph2_max
    call Spherical_cap_angle(phi_cr_max,sphere_1,sphere_2)
    phi_cr_min = 0.
    r1_sph1 = r1_sph1_min
end if

! 4. Generate a random point (phi_r,theta_r)
call random_number(u)
theta_r = 2.*pi*u

call random_number(v)
phi_r = phi_cr_min + (phi_cr_max-phi_cr_min)*v

x1_sph1 = sphere_1(1)
y1_sph1 = sphere_1(2)
z1_sph1 = sphere_1(3)

x2_sph2 = sphere_2(1)
y2_sph2 = sphere_2(2)
z2_sph2 = sphere_2(3)

x = x1_sph1 + r1_sph1*cos(theta_r)*sin(phi_r)
y = y1_sph1 + r1_sph1*sin(theta_r)*sin(phi_r)
z = z1_sph1 + r1_sph1*cos(phi_r)

! 5. Rotate that point for returning to the original coordinate system
r12 = (/ (x2_sph2-x1_sph1), (y2_sph2-y1_sph1), (z2_sph2-z1_sph1) /)
r12 = r12/norm2(r12)

v1 = (/0., 0., 1./)
v2 = (/r12(1), r12(2), r12(3)/)
s_vec = cross(v1,v2)/norm2(cross(v1,v2))
angle = acos(DOT_PRODUCT(v1,v2)/(norm2(v1)*norm2(v2)))

As = RESHAPE((/0., s_vec(3), -s_vec(2), -s_vec(3), 0., s_vec(1), s_vec(2), -s_vec(1), 0./),(/3,3/))
Identity = RESHAPE((/1., 0., 0., 0., 1., 0., 0., 0., 1./),(/3,3/))
Rot = Identity + sin(angle)*As + (1.-cos(angle))*MATMUL(As,As)

New_c = MATMUL(Rot, (/(x-x1_sph1), (y-y1_sph1), (z-z1_sph1)/))

x = x1_sph1 + New_c(1)
y = y1_sph1 + New_c(2)
z = z1_sph1 + New_c(3)
end subroutine Random_point_SC

subroutine Spherical_cap_angle(phi_cr,sphere_1,sphere_2)
implicit none
real, intent(in) :: sphere_1(4), sphere_2(4)
real, intent(out) :: phi_cr

real x0,y0,z0,r0
real x1_sph1,y1_sph1,z1_sph1,r1_sph1
real x2_sph2,y2_sph2,z2_sph2,r2_sph2
real A,B,C,D, distance, alpha_0, t, Lc_cm1, Lp_cm1
real, parameter :: pi=4.0*atan(1.0)

x1_sph1 = sphere_1(1)
y1_sph1 = sphere_1(2)
z1_sph1 = sphere_1(3)
r1_sph1 = sphere_1(4)

x2_sph2 = sphere_2(1)
y2_sph2 = sphere_2(2)
z2_sph2 = sphere_2(3)
r2_sph2 = sphere_2(4)

A = 2.*(x2_sph2-x1_sph1)
B = 2.*(y2_sph2-y1_sph1)
C = 2.*(z2_sph2-z1_sph1)
D = x1_sph1**2.0 - x2_sph2**2.0 + y1_sph1**2.0 - y2_sph2**2.0 + z1_sph1**2.0 - z2_sph2**2.0 - r1_sph1**2.0 + r2_sph2**2.0

t = (x1_sph1*A + y1_sph1*B + z1_sph1*C + D)/(A*(x1_sph1-x2_sph2) + B*(y1_sph1-y2_sph2) + C*(z1_sph1-z2_sph2))

x0 = x1_sph1 + t*(x2_sph2-x1_sph1)
y0 = y1_sph1 + t*(y2_sph2-y1_sph1)
z0 = z1_sph1 + t*(z2_sph2-z1_sph1)

distance = sqrt((x2_sph2-x1_sph1)**2. + (y2_sph2-y1_sph1)**2. + (z2_sph2-z1_sph1)**2.)

alpha_0 = acos((r1_sph1**2. + distance**2. - r2_sph2**2.)/(2.*r1_sph1*distance))
r0 = r1_sph1*sin(alpha_0)

! 3. Obtain the critical angle phi_cr
Lc_cm1 = sqrt((t*(x2_sph2-x1_sph1))**2. + (t*(y2_sph2-y1_sph1))**2. + (t*(z2_sph2-z1_sph1))**2.)

if (t .GT. 0.) then
    Lp_cm1 = sqrt(Lc_cm1**2. + r0**2.)
else if (t .LT. 0.) then
    Lp_cm1 = -sqrt(Lc_cm1**2. + r0**2.)
end if

phi_cr = acos(Lc_cm1/Lp_cm1)
end subroutine Spherical_cap_angle

subroutine CCA_Overlapping_check(Cov_max, X1,Y1,Z1,R1,n1, X2,Y2,Z2,R2,n2)
implicit none
integer, intent(in) :: n1, n2
real, intent(in) :: X1(n1),Y1(n1),Z1(n1),R1(n1), X2(n2),Y2(n2),Z2(n2),R2(n2)
real, intent(out) :: Cov_max
real Dij, Cij
integer i,j

Cov_max = 0.

do i = 1,n1
    do j = 1,n2
        Dij = sqrt((X1(i)-X2(j))**2. + (Y1(i)-Y2(j))**2. + (Z1(i)-Z2(j))**2.)
        if (Dij .LT. (R1(i)+R2(j))) then
            Cij =  ((R1(i)+R2(j))-Dij)/(R1(i)+R2(j))
                if (Cij .GT. Cov_max) then
                    Cov_max = Cij
                end if
        end if
    end do
end do

end subroutine CCA_Overlapping_check

subroutine CCA_Sticking_process_v1_reintento(theta_a,vec_0,i_vec,j_vec,X2,Y2,Z2,n2,prev_cand2,CM2)
implicit none
integer, intent(in) :: n2, prev_cand2
real, intent(in) :: vec_0(4),i_vec(3),j_vec(3), CM2(3)
real, intent(inout) :: X2(n2),Y2(n2),Z2(n2)
real, intent(out) :: theta_a

real, parameter :: pi=4.0*atan(1.0)
real x,y,z, x0,y0,z0,r0, X_cm2,Y_cm2,Z_cm2
real :: X2_new(n2),Y2_new(n2),Z2_new(n2)
real v1(3), v2(3), s_vec(3), As(3,3), Identity(3,3), Rot(3,3), New_c(3)
real angle, u
integer i

X_cm2 = CM2(1)
Y_cm2 = CM2(2)
Z_cm2 = CM2(3)

x0 = vec_0(1)
y0 = vec_0(2)
z0 = vec_0(3)
r0 = vec_0(4)

X2_new = X2
Y2_new = Y2
Z2_new = Z2

CALL RANDOM_NUMBER(u)

theta_a = 2.*pi*u

! parametrized
x = x0 + r0*cos(theta_a)*i_vec(1) + r0*sin(theta_a)*j_vec(1)
y = y0 + r0*cos(theta_a)*i_vec(2) + r0*sin(theta_a)*j_vec(2)
z = z0 + r0*cos(theta_a)*i_vec(3) + r0*sin(theta_a)*j_vec(3)

! Update coordinates of PP belonging to aggregate 2
! Based on Euler-Rodriguez formula

v1 = (/(X2_new(prev_cand2)-X_cm2), (Y2_new(prev_cand2)-Y_cm2), (Z2_new(prev_cand2)-Z_cm2)/)
v2 = (/(x-X_cm2), (y-Y_cm2), (z-Z_cm2)/)
s_vec = cross(v1,v2)/norm2(cross(v1,v2))

!this ratio might present some numerical issues
!angle = acos(DOT_PRODUCT(v1,v2)/(norm2(v1)*norm2(v2)))
if (((DOT_PRODUCT(v1,v2)/(norm2(v1)*norm2(v2))) .GT. 1.) .OR. ((DOT_PRODUCT(v1,v2)/(norm2(v1)*norm2(v2))) .LT. -1.)) then
angle = acos(1.0)
else
angle = acos(DOT_PRODUCT(v1,v2)/(norm2(v1)*norm2(v2)))
end if

As = RESHAPE((/0., s_vec(3), -s_vec(2), -s_vec(3), 0., s_vec(1), s_vec(2), -s_vec(1), 0./),(/3,3/))
Identity = RESHAPE((/1., 0., 0., 0., 1., 0., 0., 0., 1./),(/3,3/))
Rot = Identity + sin(angle)*As + (1.-cos(angle))*MATMUL(As,As)

do i = 1,size(X2_new)
    New_c = MATMUL(Rot,(/(X2_new(i)-x_cm2), (Y2_new(i)-y_cm2), (Z2_new(i)-z_cm2)/))

    X2_new(i) = X_cm2 + New_c(1)
    Y2_new(i) = Y_cm2 + New_c(2)
    Z2_new(i) = Z_cm2 + New_c(3)
end do

X2 = X2_new
Y2 = Y2_new
Z2 = Z2_new
end subroutine CCA_Sticking_process_v1_reintento

end module CCA_module
