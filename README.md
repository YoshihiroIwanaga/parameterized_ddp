# Parameterized iterative Linear Quadratic Regulator (P-iLQR) and Differential Dynamic Programming (P-DDP)

This repository contains an implementation of the iterative Linear Quadratic Regulator (iLQR) and Differential Dynamic Programming (DDP). 
The computational cost is reduced by introducing a parametric representation of the control inputs, such as zero-order, linear, and cubic interpolations. 

## Motivation

The iLQR and DDP are widely used for solving nonlinear optimal control problems [1]. 
To decrease the computational cost, we reduce the dimensionality of the decision variables by introducing a parametric representation of the control inputs. 
In this framework, only the control inputs at specific knot points are optimized, while the remaining control inputs are determined by interpolation functions (zero-order, linear, and cubic).

## Installation
-------------
To use this repository, simply clone it. You'll need to install the Numpy package using:
```
pip install numpy
```

## Examples
To run the car trajectory planning example, use the following command:
```
./scripts/run_car_traj_plan.bash
```
In this example, the upper and lower limits of vehicle velocity and steering angle are managed using the Discrete Barrier State framework [2].  

The animation below shows the optimization progress. 
The left panel displays the xy-path, while the panels on the right show the time series of states and control inputs. 
The dotted red line represents the original iLQR, and the blue line represents the P-iLQR. 
As shown, P-iLQR reaches the optimal solution faster than the original iLQR.
 

## Citations
[1] L. Weiwei and T. Emanuel, “Iterative linear quadratic regulator design
for nonlinear biological movement systems,” Proceedings of the 1st international conference on informatics in control, automation and robotics,
(ICINCO 2004), 1, 2004, pp. 222-229.  
[2] H. Almubarak, K. Stachowicz, N. Sadegh and E. A. Theodorou, ”Safety
Embedded Differential Dynamic Programming Using Discrete Barrier
States,” in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp.
2755-2762, April 2022, doi: 10.1109/LRA.2022.3143301.



