! THIS IS THE MAIN MODULE OF PC aggregation

module PCA_cca
   implicit none
contains

   subroutine PCA(Data_new, not_able_pca, Number, Mass, Radius, Df, kf, tol_ov)
      implicit none
      integer, intent(in) :: Number
      real, intent(in) :: Mass(Number), Radius(Number)
      real, intent(in) :: Df, kf, tol_ov
      logical, intent(out):: not_able_pca
      real, allocatable, intent(out):: Data_new(:, :)

      real m1, m2, m3, rg1, rg2, rg3
      real x_cm, y_cm, z_cm
      real Gamma_pc, Rmax, Cov_max
      real X_k, Y_k, Z_k, R_k, theta_a, x0, y0, z0, r0, X_sel, Y_sel, Z_sel, R_sel
      real i_vec(3), j_vec(3)
      integer n1, n2, n3
      integer previous_candidate, selected_real, intento, lista_suma
      integer, allocatable :: list(:), considerados(:)
      logical :: Gamma_real
      integer i, k, N
      real M(Number), R(Number)
      real X(Number), Y(Number), Z(Number)

      N = Number
      M = Mass
      R = Radius

      ! STEP 1: First two primary particles
      call First_two_monomers(X, Y, Z, R, M, N, m1, n1, rg1, x_cm, y_cm, z_cm, Df, kf)

      not_able_pca = .FALSE.

      if (N .GT. 2) then
         k = 3

         do while ((k .LE. N))
            ! Aggregate 1 - growing aggregate (n1, m1, rg1)
            ! Aggregate 2 - monomer to be aggregated
            n2 = 1
            m2 = M(k)
            rg2 = sqrt(0.6)*R(k)

            ! Aggregate 3 - Resulting aggregate (aggregate3 = aggregate1+aggregate2)
            n3 = n1 + n2
            m3 = m1 + m2
            rg3 = (exp(sum(log(R))/size(log(R))))*(real(n3)/kf)**(1./Df)

            call Gamma_calculation(rg1, rg2, rg3, n1, n2, n3, Gamma_real, Gamma_pc)

            allocate (considerados(N - k + 1))      ! list of non-aggregated monomers
            considerados(1:size(considerados)) = 0
            considerados(1) = 1                ! which mean that k has been considered

            ! Step 3: List of candidates (candidates belonging to aggregate_1) to be aggregated with k
            allocate (list(n1))
            call Random_select_list(X, Y, Z, R, N, list, Rmax, n1, x_cm, y_cm, z_cm, Gamma_pc, Gamma_real)

            lista_suma = 0

            do while (lista_suma .EQ. 0)
               ! Step 2: select a new PP (to be aggregated)
               do while ((sum(list) .EQ. 0) .and. (sum(considerados) .LE. (N - k)))
             call Search_list(X,Y,Z,R,M,N,k,Df,kf,list,Rmax,m2,rg2,m3,rg3,Gamma_pc,Gamma_real,n1,rg1,n3,considerados,x_cm,y_cm,z_cm)
               end do

               previous_candidate = 0 !there are no candidates (belonging to aggregate 1) previously considered

               ! Step 4: Select a candidate
               if ((sum(list) .GT. 0)) then
                  call Random_select_list_pick_one(selected_real, list, previous_candidate)
                  previous_candidate = selected_real
               elseif ((sum(considerados) .EQ. N - k + 1)) then
                  not_able_pca = .true.
                  RETURN
               end if

               intento = 1

               ! Step 5: sticking process
               X_sel = X(selected_real)
               Y_sel = Y(selected_real)
               Z_sel = Z(selected_real)
               R_sel = R(selected_real)
               R_k = R(k)
        call sticking_process(X_k,Y_k,Z_k,theta_a,x0,y0,z0,r0,i_vec,j_vec, X_sel, Y_sel, Z_sel, R_sel, R_k, x_cm,y_cm,z_cm,Gamma_pc)
               X(k) = X_k
               Y(k) = Y_k
               Z(k) = Z_k

               ! Overlaping check
               call Overlapping_check(Cov_max, X(1:k), Y(1:k), Z(1:k), R(1:k), k)

               do while ((Cov_max .GT. tol_ov) .AND. (intento .LT. 360))
                  ! Step 6: Rotation
                  call Sticking_process_reintento(X_k, Y_k, Z_k, x0, y0, z0, r0, i_vec, j_vec, theta_a)
                  X(k) = X_k
                  Y(k) = Y_k
                  Z(k) = Z_k

                  call Overlapping_check(Cov_max, X(1:k), Y(1:k), Z(1:k), R(1:k), k)
                  intento = intento + 1

! Have we done more than int(360) rotations?
! Are there any candidates available?
                  if ((mod(real(intento), real((359))) .EQ. 0.) .AND. (sum(list) .GT. 1)) then
! Step 4: Select a candidate
                     call Random_select_list_pick_one(selected_real, list, previous_candidate)

                     X_sel = X(selected_real)
                     Y_sel = Y(selected_real)
                     Z_sel = Z(selected_real)
                     R_sel = R(selected_real)
                     R_k = R(k)
     call sticking_process(X_k, Y_k, Z_k, theta_a,x0,y0,z0,r0,i_vec,j_vec, X_sel, Y_sel, Z_sel, R_sel, R_k, x_cm,y_cm,z_cm,Gamma_pc)
                     X(k) = X_k
                     Y(k) = Y_k
                     Z(k) = Z_k

                     previous_candidate = selected_real
                     intento = 1
                     call Overlapping_check(Cov_max, X(1:k), Y(1:k), Z(1:k), R(1:k), k)
                  end if
               end do

               lista_suma = sum(list)

               if (Cov_max .GT. tol_ov) then
                  lista_suma = 0
                  list = list*0
               end if

            end do

! cumulated parameters
            x_cm = (x_cm*m1 + X(k)*m2)/(m1 + m2)
            y_cm = (y_cm*m1 + Y(k)*m2)/(m1 + m2)
            z_cm = (z_cm*m1 + Z(k)*m2)/(m1 + m2)

            n1 = n3
            m1 = m3
            rg1 = (exp(sum(log(R))/size(log(R))))*(real(n1)/kf)**(1./Df)
            k = k + 1
            deallocate (list)
            deallocate (considerados)
         end do
      end if

      allocate (Data_new(N, 4))
      if (not_able_pca) then
         return
      else
         do i = 1, N
            Data_new(i, :) = (/X(i), Y(i), Z(i), R(i)/)
         end do
      end if

   end subroutine PCA

   subroutine First_two_monomers(X, Y, Z, R, M, N, m1, n1, rg1, x_cm, y_cm, z_cm, Df, kf)
      implicit none
      integer, intent(in) :: N
      real, intent(in) :: R(N), M(N)
      real, intent(in) :: Df, kf
      real, intent(out) :: X(N), Y(N), Z(N)
      real, intent(out) :: m1, rg1
      real, intent(out) :: x_cm, y_cm, z_cm
      integer, intent(out) :: n1
      real theta, phi

      X(1:N) = 0.
      Y(1:N) = 0.
      Z(1:N) = 0.

      call Random_point_sphere(theta, phi)

      X(2) = X(1) + (R(1) + R(2))*cos(theta)*sin(phi)
      Y(2) = Y(1) + (R(1) + R(2))*sin(theta)*sin(phi)
      Z(2) = Z(1) + (R(1) + R(2))*cos(phi)

      m1 = M(1) + M(2)
      n1 = 2

      rg1 = (exp(sum(log((/R(1), R(2)/)))/2))*(real(n1)/kf)**(1./Df)

      x_cm = (X(1)*M(1) + X(2)*M(2))/(M(1) + M(2))
      y_cm = (Y(1)*M(1) + Y(2)*M(2))/(M(1) + M(2))
      z_cm = (Z(1)*M(1) + Z(2)*M(2))/(M(1) + M(2))

   end subroutine First_two_monomers

   subroutine Random_point_sphere(theta, phi)
      implicit none
      real, parameter :: pi = 4.0*atan(1.0)
      real u, v, theta, phi

      CALL RANDOM_NUMBER(u)
      CALL RANDOM_NUMBER(v)

      theta = 2.*pi*u
      phi = acos(2.*v - 1.)
   end subroutine Random_point_sphere

   subroutine Gamma_calculation(rg1, rg2, rg3, n1, n2, n3, Gamma_real, Gamma_pc)
      real, intent(in) :: rg1, rg2, rg3
      integer, intent(in):: n1, n2, n3
      real, intent(inout) :: Gamma_pc
      logical, intent(out) :: Gamma_real
      real rg3_auxiliar

      if (rg3 .LT. rg1) then
         rg3_auxiliar = rg1
      else
         rg3_auxiliar = rg3
      end if

! Step 2: k is the new PP
      if ((real(n3)**2.)*(rg3_auxiliar**2.) .GT. (real(n3)*(real(n1)*rg1**2.+real(n2)*rg2**2.))) then
         Gamma_pc = sqrt(((real(n3)**2.)*(rg3_auxiliar**2.) - real(n3)*(real(n1)*rg1**2.+real(n2)*rg2**2.))/(real(n1)*real(n2)))
         Gamma_real = .true.
      else
         Gamma_real = .false.
      end if

   end subroutine Gamma_calculation

   subroutine Random_select_list(X, Y, Z, R, N, list, Rmax, n1, x_cm, y_cm, z_cm, Gamma_pc, Gamma_real)
      implicit none
      integer, intent(in) :: N
      real, intent(in) :: X(N), Y(N), Z(N), R(N)
      real, intent(in) :: x_cm, y_cm, z_cm, Gamma_pc
      integer, intent(in) :: n1
      logical, intent(in) :: Gamma_real
      integer, intent(out) :: list(n1)
      real, intent(out) :: Rmax
      integer ii
      real dist

      Rmax = 0.
      list(1:n1) = 0

      if (Gamma_real) then
         do ii = 1, n1
            dist = sqrt((X(ii) - x_cm)**2.+(Y(ii) - y_cm)**2.+(Z(ii) - z_cm)**2.)

            if (dist .GT. Rmax) then !only to save Rmax
               Rmax = dist
            end if

            if ((dist .GT. (Gamma_pc - R(n1 + 1) - R(ii))) .and. (dist .LE. (Gamma_pc + R(n1 + 1) + R(ii)))) then
               list(ii) = 1
            end if

            if ((R(n1 + 1) + R(ii)) .GT. Gamma_pc) THEN
               list(ii) = 0
            end if
         end do
      else
         return
      end if
   end subroutine Random_select_list

   subroutine Search_list(X,Y,Z,R,M,N,k,Df,kf,list,Rmax,m2,rg2,m3,rg3,Gamma_pc,Gamma_real,n1,rg1,n3,considerados,x_cm,y_cm,z_cm)
      implicit none
      integer, intent(in) :: N, k
      real, intent(in) :: X(N), Y(N), Z(N)
      real, intent(inout) :: R(N), M(N)
      real, intent(in) :: rg1, x_cm, y_cm, z_cm, Df, kf
      integer, intent(in) :: n1, n3
      real, intent(out) :: Gamma_pc, m2, rg2, m3, rg3
      integer, intent(inout), allocatable :: list(:)
      integer, intent(inout), allocatable :: considerados(:)
      real, intent(inout) :: Rmax
      logical, intent(inout) :: Gamma_real
      integer, allocatable :: vector_search2(:)
      real R_sl(N), M_sl(N)
      real rand_n
      integer vector_search(N - k + 1)
      integer RS_uno, ii

      R_sl = R
      M_sl = M

! it should go from "k" to "N"
      do ii = 1, size(vector_search)
         vector_search(ii) = ii + k - 1
      end do

      do ii = 1, size(considerados)
         if (considerados(ii) .EQ. 1) then
            vector_search(ii) = 0
         end if
      end do

      allocate (vector_search2(size(pack(vector_search, vector_search .NE. 0))))
      vector_search2 = pack(vector_search, vector_search .NE. 0)

! Random sample from 'vector_search2'
      if (size(vector_search2) .GT. 1) then
         call RANDOM_NUMBER(rand_n)
         RS_uno = vector_search2(1 + INT((REAL(size(vector_search2) - 1))*rand_n))
      else
         RS_uno = vector_search2(1)
      end if

!interchange mass and radius
      R(RS_uno) = R_sl(k)
      R(k) = R_sl(RS_uno)

      M(RS_uno) = M_sl(k)
      M(k) = M_sl(RS_uno)

! update aggregate
      m2 = M(k)
      rg2 = sqrt(0.6)*R(k)

      m3 = sum(M(1:k))
      rg3 = (exp(sum(log(R))/size(log(R))))*(real(n3)/kf)**(1./Df)

      call Gamma_calculation(rg1, rg2, rg3, n1, 1, n3, Gamma_real, Gamma_pc)

      call Random_select_list(X, Y, Z, R, N, list, Rmax, n1, x_cm, y_cm, z_cm, Gamma_pc, Gamma_real)
      considerados(RS_uno - k + 1) = 1

   end subroutine Search_list

   subroutine Random_select_list_pick_one(selected_real, list, previous_candidate)
      implicit none
      integer, intent(in) :: previous_candidate
      integer, intent(inout), allocatable :: list(:)
      integer, intent(out) :: selected_real
      real rand_n
      integer, allocatable :: list2(:)
      integer selected, jj, ii

! Randomly select a new candidate
      if (previous_candidate .GT. 0) then
         list(previous_candidate) = 0
      end if

! random selection of a monomer (Uniformly distributed)
      allocate (list2(size(pack(list, list .GT. 0))))
      list2 = pack(list, list .GT. 0)

      call RANDOM_NUMBER(rand_n)
      selected = 1 + INT((REAL(size(list2) - 1))*rand_n)

      jj = 0
      do ii = 1, size(list)
         if (list(ii) .GT. 0) then
            jj = jj + 1
         end if

         if (jj .EQ. selected) then
            selected_real = ii
            return
         end if
      end do
   end subroutine Random_select_list_pick_one

subroutine sticking_process(X_k, Y_k, Z_k, theta_a,x0,y0,z0,r0,i_vec,j_vec, X_sel, Y_sel, Z_sel, R_sel, R_k,x_cm,y_cm,z_cm,Gamma_pc)
      implicit none
      real, intent(in) :: X_sel, Y_sel, Z_sel, R_sel, R_k
      real, intent(in) :: x_cm, y_cm, z_cm, Gamma_pc
      real, intent(out) :: X_k, Y_k, Z_k
      real, intent(out) :: theta_a, x0, y0, z0, r0
      real, intent(out) :: i_vec(3), j_vec(3)
      real k_vec(3)
      real X1, Y1, Z1, R1, X2, Y2, Z2, R2
      real A, B, C, D, t_sp, alpha_0, phi, distanc, AmBdC

! To understand this function please refer to Appenix A (Sphere-sphere intersection) of the article
! Parameters
! (x1 y1, z1) -> center sphere 1: SELECTED (from growing aggregate 1), radius  R_sel + R_k
! (z2, y2, z2) -> center sphere 2: Ceter of mass of growing aggregate 1, radius -> Gamma_pc

      X1 = X_sel
      Y1 = Y_sel
      Z1 = Z_sel
      R1 = R_sel + R_k

      X2 = x_cm
      Y2 = y_cm
      Z2 = z_cm
      R2 = Gamma_pc

! plane intersection both spheres: Ax + By Cz + D = 0
      A = 2.*(x2 - x1)
      B = 2.*(y2 - y1)
      C = 2.*(z2 - z1)
      D = x1**2.-x2**2.+y1**2.-y2**2.+z1**2.-z2**2.-r1**2.+r2**2.

      t_sp = (x1*A + y1*B + z1*C + D)/(A*(x1 - x2) + B*(y1 - y2) + C*(z1 - z2))

      x0 = x1 + t_sp*(x2 - x1)
      y0 = y1 + t_sp*(y2 - y1)
      z0 = z1 + t_sp*(z2 - z1)

      distanc = sqrt((x2 - x1)**2.+(y2 - y1)**2.+(z2 - z1)**2.)

      alpha_0 = acos((r1**2.+distanc**2.-r2**2.)/(2.*r1*distanc))
      r0 = r1*sin(alpha_0)

      AmBdC = (A + B)/C

      k_vec = (/A, B, C/)/(sqrt(A**2.+B**2.+C**2.))               !vector perpendicular to plane Ax+By+Cz+D=0
      i_vec = (/1., 1., -AmBdC/)/(sqrt(1.+1.+(AmBdC)**2.)) !I've set x=1 and y=1 and solved Ax+By+Cz=0 to find z perpendicular
      j_vec = cross(k_vec, i_vec)

      call Random_point_sphere(theta_a, phi)

! parametrized
      x_k = x0 + r0*cos(theta_a)*i_vec(1) + r0*sin(theta_a)*j_vec(1)
      y_k = y0 + r0*cos(theta_a)*i_vec(2) + r0*sin(theta_a)*j_vec(2)
      z_k = z0 + r0*cos(theta_a)*i_vec(3) + r0*sin(theta_a)*j_vec(3)

   end subroutine sticking_process

   FUNCTION cross(a, b)
      REAL, DIMENSION(3) :: cross
      REAL, DIMENSION(3), INTENT(IN) :: a, b

      cross(1) = a(2)*b(3) - a(3)*b(2)
      cross(2) = a(3)*b(1) - a(1)*b(3)
      cross(3) = a(1)*b(2) - a(2)*b(1)
   END FUNCTION cross

   subroutine Overlapping_check(Cov_max, X_con, Y_con, Z_con, R_con, k)
      implicit none
      integer, intent(in) :: k
      real, intent(in) :: X_con(k), Y_con(k), Z_con(k), R_con(k)
      real, intent(out) :: Cov_max
      real C(k - 1)
      real distanc_kj
      integer jj

! 1. Distance between monomers
      do jj = 1, (k - 1)
         distanc_kj = sqrt((x_con(k) - x_con(jj))**2.+(y_con(k) - y_con(jj))**2.+(z_con(k) - z_con(jj))**2.)

         if (distanc_kj .LT. (R_con(k) + R_con(jj))) then
            C(jj) = ((R_con(k) + R_con(jj)) - distanc_kj)/(R_con(k) + R_con(jj))
         else
            C(jj) = 0.
         end if
      end do

      Cov_max = maxval(C)
   end subroutine Overlapping_check

   subroutine Sticking_process_reintento(X_k, Y_k, Z_k, x0, y0, z0, r0, i_vec, j_vec, theta_a)
      implicit none
      real, intent(in) :: i_vec(3), j_vec(3)
      real, intent(in) :: x0, y0, z0, r0
      real, intent(out) :: X_k, Y_k, Z_k
      real, intent(out) :: theta_a
      real, parameter :: pi = 4.0*atan(1.0)
      real u

      CALL RANDOM_NUMBER(u)

      theta_a = 2.*pi*u

      x_k = x0 + r0*cos(theta_a)*i_vec(1) + r0*sin(theta_a)*j_vec(1)
      y_k = y0 + r0*cos(theta_a)*i_vec(2) + r0*sin(theta_a)*j_vec(2)
      z_k = z0 + r0*cos(theta_a)*i_vec(3) + r0*sin(theta_a)*j_vec(3)

   end subroutine Sticking_process_reintento

end module PCA_cca
