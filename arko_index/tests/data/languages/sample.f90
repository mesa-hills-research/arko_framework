! Numerical helpers for vectors.
module vectors
  implicit none
  private
  public :: norm, normalize, vec3

  type :: vec3
    real :: x, y, z
  contains
    procedure :: length => vec3_length
  end type vec3

  interface norm
    module procedure norm_real, norm_vec3
  end interface norm

contains

  ! The Euclidean norm of an array.
  pure function norm_real(v) result(n)
    real, intent(in) :: v(:)
    real :: n
    n = sqrt(sum(v**2))
  end function norm_real

  pure function norm_vec3(v) result(n)
    type(vec3), intent(in) :: v
    real :: n
    n = sqrt(v%x**2 + v%y**2 + v%z**2)
  end function norm_vec3

  real function vec3_length(self)
    class(vec3), intent(in) :: self
    vec3_length = norm_vec3(self)
  end function vec3_length

  subroutine normalize(v)
    real, intent(inout) :: v(:)
    real :: n
    n = norm_real(v)
    if (n > 0.0) then
      v = v / n
    end if
  end subroutine normalize

end module vectors

program main
  use vectors
  implicit none
  real :: v(3) = [3.0, 4.0, 0.0]
  call normalize(v)
  print *, v
end program main
