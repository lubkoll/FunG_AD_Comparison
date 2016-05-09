// Copyright (C) 2015 by Lars Lubkoll. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#include <chrono>
#include <iostream>

#include <fadiff.h>
#include <badiff.h>

#include "Sacado.hpp"

#include <adept.h>

#include <cppad/cppad.hpp>

#define TAPELESS
#include "adolc/adtl.h"

#include "fung/fung.hh"

#include "example.hh"


template <class Time>
void printMeasurement(double f, double dfdx, Time startTime)
{
  using std::cout;
  using std::endl;
  using namespace std::chrono;
  cout << "computation time: " << duration_cast<milliseconds>(high_resolution_clock::now()-startTime).count()/1000. << "s\n";
  cout << "function value  : " << f << endl;
  cout << "first derivative: " << dfdx << endl << endl;
}

template <class Scalar, class Time>
void beforeMeasurement(Scalar& x, Time& startTime, std::string adName)
{
  std::cout << adName << std::endl;
  x = 1;
  startTime = std::chrono::high_resolution_clock::now();
}

int main()
{
  using std::cout;
  using std::endl;
  using namespace std::chrono;

  auto iter = 10'000'000u;
  auto startTime = high_resolution_clock::now();
  double x = 5,
         f = 0,
         dfdx = 0,
         scaling = 1.00000001;

  constexpr auto nVars = 1;
  constexpr auto id = 0;

  cout << "Function: " << functionAsString() << endl;
  cout << "#Evaluations: " << iter << endl << endl;


  FADBADForward<TestFunction> f_fadbad_forward;
  beforeMeasurement(x,startTime,"FADBAD++ (forward)");
  for(auto i=0u; i<iter; ++i)
  {
    f = f_fadbad_forward(dfdx,x);
    x *= scaling;
  }
  printMeasurement(f,dfdx,startTime);


  FADBADBackward<TestFunction> f_fadbad_backward;
  beforeMeasurement(x,startTime,"FADBAD++ (backward)");
  for(auto i=0u; i<iter; ++i)
  {
    f = f_fadbad_backward(dfdx,x);
    x *= scaling;
  }
  printMeasurement(f,dfdx,startTime);


  auto r_fad = Sacado::Fad::SFad<double,nVars>(nVars,id,0),
       x_fad = Sacado::Fad::SFad<double,nVars>(nVars,id,x);
  beforeMeasurement(x_fad.val(),startTime,"SACADO (FAD)");
  for(auto i=0u; i<iter; ++i){
    r_fad = testFunctionWithFixedType(x_fad);
    x_fad.val() *= scaling;
  }
  printMeasurement(r_fad.val(),r_fad.dx(id),startTime);


  auto r_elrfad = Sacado::ELRFad::DFad<double>(nVars,id,0),
       x_elrfad = Sacado::ELRFad::DFad<double>(nVars,id,x);
  beforeMeasurement(x_elrfad.val(),startTime,"SACADO (ELRFAD)");
  for(auto i=0u; i<iter; ++i){
    r_elrfad = testFunctionWithFixedType(x_elrfad);
    x_elrfad.val() *= scaling;
  }
  printMeasurement(r_elrfad.val(),r_elrfad.dx(id),startTime);


  adept::Stack stack;
  adept::adouble ax, y;
  beforeMeasurement(ax,startTime,"ADEPT");
  for(auto i=0u; i<iter; ++i)
  {
    stack.new_recording();
    y = testFunctionWithFixedType(ax);
    y.set_gradient(1.);
    stack.compute_adjoint();
    f = y.value();
    dfdx = ax.get_gradient();
    ax *= scaling;
  }
  printMeasurement(f,dfdx,startTime);

    
  adtl::adouble ax1;
  const double ax_init[] = {1.};
  adtl::adouble ay1;
  beforeMeasurement(ax1,startTime,"ADOLC");
  for(auto i=0u; i<iter; ++i)
  {
    ax1.setADValue(ax_init);
    ay1 = testFunctionWithFixedType(ax1);
    f = ay1.getValue();
    dfdx = ay1.getADValue()[0];
    ax1 *= scaling;
  }
  printMeasurement(f,dfdx,startTime);
  

  auto f_cppad = cppADFunction(x);
  f_cppad.optimize();
  std::vector<double> X0(nVars,x);
  beforeMeasurement(X0[id],startTime,"CPPAD");
  for( auto i = 0u ; i < iter ; ++i )
  {
    f = f_cppad.Forward(0,X0)[id];
    dfdx = f_cppad.Jacobian(X0)[id];
    X0[id] *= scaling;
  }
  printMeasurement(f,dfdx,startTime);


  auto f_fung = FunG::finalize( testFunction(FunG::variable<id>(1.)) );
  beforeMeasurement(x,startTime,"FunG");
  for(auto i=0u; i<iter; ++i)
  {
    f_fung.update<id>(x);
    f = f_fung();
    dfdx = f_fung.d1<id>(1.);
    x *= scaling;
  }
  printMeasurement(f,dfdx,startTime);


  TestFunction function;
  beforeMeasurement(x,startTime,"Manual");
  for(auto i=0u; i<iter; ++i)
  {
    f = function(x);
    dfdx = function.d1(x);
    x *= scaling;
  }
  printMeasurement(f,dfdx,startTime);


  CachingTestFunction<> optfunction;
  beforeMeasurement(x,startTime,"Manual (Caching)");
  for(auto i=0u; i<iter; ++i)
  {
    f = optfunction(x);
    dfdx = optfunction.d1(x);
    x *= scaling;
  }
  printMeasurement(f,dfdx,startTime);
}
