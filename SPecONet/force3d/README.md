# User instructions

Program that solves the incompressible Navier Stokes equations for given forceing functions as inputs on a 3D square domain using SPecONet.

To train SPecOnet, execute the code, total2.py.  


After the code has finished running, The computational data are saved as follows:
- The file named "3dforce100sigma5all.csv" records Rel. $L^2$ error for test samples at $`t=k\Delta t`$, where $`\Delta t=0.01`$, $`k=1,2,\cdots, 100`$.  
- In training/NS3d1.0/uexsigma5, the $L^2$ norm of reference solutions regarding test samples for each time, $`\|u\|_{L^2}^2`$, $`\|v\|_{L^2}^2`$, and $`\|\nabla p\|_{L^2}^2`$ are saved. 
- In training/NS3d1.0/pp, reference solution and inferences of $p$ regarding test samples for each time are saved. 

  





