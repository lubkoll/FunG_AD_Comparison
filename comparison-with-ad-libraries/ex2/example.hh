#ifndef EXAMPLE_HH
#define EXAMPLE_HH

#include <cmath>
#include <vector>


/// Test function for Sacado and Adept. Both rely on special types and don't work well with auto.
template <class T>
T testFunctionWithFixedType(const T& x)
{
  T z = exp(sqrt(x))+1;
  return cos(x)+sin(z);
}


template <class T>
auto testFunction(const T& x)
{
  auto z = exp(sqrt(x))+1;
  return cos(x)+sin(z);
}


struct TestFunction
{
  template <class T>
  auto operator()(const T& x) const
  {
    return testFunction(x);
  }

  template <class T>
  auto d1(const T& x) const
  {
    auto z0 = sqrt(x);
    auto z = exp(z0);
    auto dz = 0.5*z/z0;
    return -sin(x) + cos(z+1)*dz;
  }
};


template <class T=double>
struct CachingTestFunction
{
  T operator()(const T& x) const
  {
    z0 = sqrt(x);
    z = exp(z0);
    dz = 0.5*z/z0;
    return cos(x) + sin(z+1);
  }

  T d1(const T& x) const
  {
    return -sin(x) + cos(z+1)*dz;
  }
  mutable T z, z0, dz;
};


/// Interface for CppAD.
auto cppADFunction(double x)
{
  using CppAD::AD;

  std::vector< AD<double> > X(1);
  X[0] = x;

  CppAD::Independent(X);

  std::vector< AD<double> > Y(1);
  Y[0] = testFunction(X[0]);

  return CppAD::ADFun<double>(X, Y);
}


/// Interface for FADBAD++, forward mode.
template <class C>
struct FADBADForward
{
  template <class T>
  T operator()( T& o_dfdx, const T& i_x) const
  {
    using namespace fadbad;
    F<T,1> x(i_x);      // Initialize arguments
    x.diff(0);          // Differentiate wrt. x
    C func;             // Instantiate functor
    F<T,1> f(func(x));  // Evaluate function and record DAG
    o_dfdx=f.d(0);      // Value of df/dx
    return f.x();       // Return function value
  }
};


/// Interface for FADBAD++, backward mode
template <class C>
struct FADBADBackward
{
  template <class T>
  T operator()( T& o_dfdx, const T& i_x ) const
  {
    using namespace fadbad;
    B<T> x(i_x);        // Initialize arguments
    C func;             // Instantiate functor
    B<T> f(func(x));    // Evaluate function and record DAG
    f.diff(0,1);        // Differentiate
    o_dfdx=x.d(0);      // Value of df/dx
    return f.x();       // Return function value
  }
};


std::string functionAsString()
{
  return "cos(x) + sin( exp( sqrt(x) ) + 1 )";
}

#endif // EXAMPLE_HH
