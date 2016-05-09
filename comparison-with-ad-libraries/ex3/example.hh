#ifndef EXAMPLE_HH
#define EXAMPLE_HH

#include <cmath>
#include <vector>


/// Test function for Sacado and Adept. Both rely on special types and don't work well with auto.
template <class T>
T testFunctionWithFixedType(const T& x, const T& y, const T& z)
{
  T s = sqrt(x);
  return y*s + sin(s) + z;

}


template <class X, class Y, class Z>
auto testFunction(const X& x, const Y& y, const Z& z)
{
  auto s = sqrt(x);
  return y*s + sin(s) + z;

}


struct TestFunction
{
  template <class T>
  auto operator()(const T& x, const T& y, const T& z) const
  {
    return testFunction(x,y,z);
  }

  template <class T>
  auto d1(const T& x, const T& y, const T&, int id) const
  {
    auto s = sqrt(x);
    if( id == 0 ) return 0.5*(y+cos(s))/s;
    if( id == 1 ) return s;
    /*if( id == 2 )*/
    return 1.;
  }
};


template <class T=double>
struct CachingTestFunction
{
  T operator()(const T& x, const T& y, const T& z) const
  {
    s = sqrt(x);
    return y*s + sin(s) + z;
  }

  template <int id>
  T d1(const T&, const T& y, const T&) const
  {
    if( id == 0 ) return 0.5*(y+cos(s))/s;
    if( id == 1 ) return s;
    /*if( id == 2 )*/
    return 1;
  }
  mutable T s;
};


/// Interface for CppAD.
auto cppADFunction(double x)
{
  using CppAD::AD;

  auto nVars = 3;
  std::vector< AD<double> > X(nVars);
  X[0] = x;
  X[1] = x;
  X[2] = x;

  CppAD::Independent(X);

  std::vector< AD<double> > Y(1);
  Y[0] = testFunction(X[0],X[1],X[2]);

  return CppAD::ADFun<double>(X, Y);
}


/// Interface for FADBAD++, forward mode.
template <class Function>
struct FADBADForward
{
  template <class T>
  T operator()( T& o_dfdx, T& o_dfdy, T& o_dfdz, const T& i_x, const T& i_y, const T& i_z) const
  {
    using namespace fadbad;
    F<T,3> x(i_x), y(i_y), z(i_z);
    x.diff(0);
    y.diff(1);
    z.diff(2);
    Function func;
    F<T,3> f(func(x,y,z));
    o_dfdx = f.d(0);
    o_dfdy = f.d(1);
    o_dfdz = f.d(2);
    return f.x();
  }
};


/// Interface for FADBAD++, backward mode
template <class Function>
struct FADBADBackward
{
  template <class T>
  T operator()( T& o_dfdx, T& o_dfdy, T& o_dfdz, const T& i_x, const T& i_y, const T& i_z) const
  {
    using namespace fadbad;
    B<T> x(i_x), y(i_y), z(i_z);
    Function func;
    B<T> f(func(x,y,z));
    f.diff(0,3);
    f.diff(1,3);
    f.diff(2,3);
    o_dfdx = x.d(0);
    o_dfdy = x.d(1);
    o_dfdz = x.d(2);
    return f.x();
  }
};


std::string functionAsString()
{
  return "y*x^0.5 + sin(x^0.5) + z";
}

#endif // EXAMPLE_HH
