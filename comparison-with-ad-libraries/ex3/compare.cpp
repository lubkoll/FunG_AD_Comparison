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
#define NUMBER_DIRECTIONS 3
#include "adolc/adtl.h"

#include "fung/fung.hh"

#include "example.hh"

//adtl::setNumDir(3);

template <class Time>
void printMeasurement(double f, double dfdx, double dfdy, double dfdz, Time startTime)
{
  using std::cout;
  using std::endl;
  using namespace std::chrono;
  cout << "computation time: " << duration_cast<milliseconds>(high_resolution_clock::now()-startTime).count()/1000. << "s\n";
  cout << "function value  : " << f << endl;
  cout << "dfdx: " << dfdx << endl;
  cout << "dfdy: " << dfdy << endl;
  cout << "dfdz: " << dfdz << endl << endl;
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
         dfdx = 0, dfdy = 0, dfdz = 0,
         scaling = 1.00000001;

  constexpr auto nVars = 3;
  constexpr auto idx = 0;
  constexpr auto idy = 1;
  constexpr auto idz = 2;

  cout << "Function: " << functionAsString() << endl;
  cout << "#Evaluations: " << iter << endl << endl;


  FADBADForward<TestFunction> f_fadbad_forward;
  beforeMeasurement(x,startTime,"FADBAD++ (forward)");
  for(auto i=0u; i<iter; ++i)
  {
    f = f_fadbad_forward(dfdx,dfdy,dfdz,x,x,x);
    x *= scaling;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);

/*
  FADBADBackward<TestFunction> f_fadbad_backward;
  beforeMeasurement(x,startTime,"FADBAD++ (backward)");
  for(auto i=0u; i<iter; ++i)
  {
    f = f_fadbad_backward(dfdx,dfdy,dfdz,x,x,x);
    x *= scaling;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);
*/

  beforeMeasurement(x,startTime,"SACADO (FAD)");
  auto r_fad = Sacado::ELRFad::DFad<double>(1,0,0),
       x_fad = Sacado::ELRFad::DFad<double>(nVars,idx,x),
       y_fad = Sacado::ELRFad::DFad<double>(nVars,idy,x),
       z_fad = Sacado::ELRFad::DFad<double>(nVars,idz,x);
  for(auto i=0u; i<iter; ++i){
    r_fad = testFunctionWithFixedType(x_fad,y_fad,z_fad);
    x *= scaling;
    x_fad.val() = x;
    y_fad.val() = x;
    z_fad.val() = x;
  }
  printMeasurement(r_fad.val(),
                   r_fad.dx(idx),
                   r_fad.dx(idy),
                   r_fad.dx(idz),
                   startTime);


  beforeMeasurement(x,startTime,"SACADO (ELRFAD)");
  auto r_elrfad = Sacado::ELRFad::DFad<double>(1,0,0),
       x_elrfad = Sacado::ELRFad::DFad<double>(nVars,idx,x),
       y_elrfad = Sacado::ELRFad::DFad<double>(nVars,idy,x),
       z_elrfad = Sacado::ELRFad::DFad<double>(nVars,idz,x);
  for(auto i=0u; i<iter; ++i){
    r_elrfad = testFunctionWithFixedType(x_elrfad,y_elrfad,z_elrfad);
    x *= scaling;
    x_elrfad.val() = x;
    y_elrfad.val() = x;
    z_elrfad.val() = x;
  }
  printMeasurement(r_elrfad.val(),
                   r_elrfad.dx(idx),
                   r_elrfad.dx(idy),
                   r_elrfad.dx(idz),
                   startTime);


  adept::Stack stack;
  beforeMeasurement(x,startTime,"ADEPT");
  adept::adouble ax[3] = {x,x,x};
  adept::adouble y;
  for(auto i=0u; i<iter; ++i)
  {
    stack.new_recording();
    y = testFunctionWithFixedType(ax[0],ax[1],ax[2]);
    y.set_gradient(1.);
    stack.compute_adjoint();
    f = y.value();
    dfdx = ax[idx].get_gradient();
    dfdy = ax[idy].get_gradient();
    dfdz = ax[idz].get_gradient();
    x *= scaling;
    ax[idx] = ax[idy] = ax[idz] = x;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);

    
  dfdy = dfdz = 0;
  const double ax_init[] = {1.}, axz_init[] = {0.};
  adtl::adouble ay1;
  beforeMeasurement(x,startTime,"ADOLC");
  adtl::adouble ax1[3] = {x,x,x};
  for(auto i=0u; i<iter; ++i)
  {
    ax1[idx].setADValue(ax_init);
    ax1[idy].setADValue(axz_init);
    ax1[idz].setADValue(axz_init);
    ay1 = testFunctionWithFixedType(ax1[idx],ax1[idy],ax1[idz]);
    f = ay1.getValue();
    dfdx = ay1.getADValue(0);
    
    ax1[idx].setADValue(axz_init);
    ax1[idy].setADValue(ax_init);
    ax1[idz].setADValue(axz_init);
    ay1 = testFunctionWithFixedType(ax1[idx],ax1[idy],ax1[idz]);
    dfdy = ay1.getADValue(0);
    
    ax1[idx].setADValue(axz_init);
    ax1[idy].setADValue(axz_init);
    ax1[idz].setADValue(ax_init);
    ay1 = testFunctionWithFixedType(ax1[idx],ax1[idy],ax1[idz]);
    dfdz = ay1.getADValue(0);
    
    x *= scaling;
    ax1[idx] = ax1[idy] = ax1[idz] = x;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);


  auto f_cppad = cppADFunction(x);
  f_cppad.optimize();
  beforeMeasurement(x,startTime,"CPPAD");
  std::vector<double> X0(nVars,x);
  for( auto i = 0u ; i < iter ; ++i )
  {
    f = f_cppad.Forward(0,X0)[0];
    dfdx = f_cppad.Jacobian(X0)[idx];
    dfdy = f_cppad.Jacobian(X0)[idy];
    dfdz = f_cppad.Jacobian(X0)[idz];
    X0[idx] *= scaling;
    X0[idy] *= scaling;
    X0[idz] *= scaling;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);

  using FunG::variable;
  auto f_fung = FunG::finalize( testFunction( variable<idx>(1.) , variable<idy>(1.) , variable<idz>(1.) ) );
  beforeMeasurement(x,startTime,"FunG");
  for(auto i=0u; i<iter; ++i)
  {
    f_fung.update<idx>(x);
    f_fung.update<idy>(x);
    f_fung.update<idz>(x);
    f = f_fung();
    dfdx = f_fung.d1<idx>(1.);
    dfdy = f_fung.d1<idy>(1.);
    dfdz = f_fung.d1<idz>(1.);
    x *= scaling;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);


  TestFunction function;
  beforeMeasurement(x,startTime,"Manual");
  for(auto i=0u; i<iter; ++i)
  {
    f = function(x,x,x);
    dfdx = function.d1(x,x,x,0);
    dfdy = function.d1(x,x,x,1);
    dfdz = function.d1(x,x,x,2);
    x *= scaling;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);


  CachingTestFunction<> optfunction;
  beforeMeasurement(x,startTime,"Manual (Caching)");
  for(auto i=0u; i<iter; ++i)
  {
    f = optfunction(x,x,x);
    dfdx = optfunction.d1<0>(x,x,x);
    dfdy = optfunction.d1<1>(x,x,x);
    dfdz = optfunction.d1<2>(x,x,x);
    x *= scaling;
  }
  printMeasurement(f,dfdx,dfdy,dfdz,startTime);
}
