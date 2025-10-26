# User instructions

Program that solves the incompressible Navier Stokes equations for given forceing functions as inputs on a 2D square domain using SPecONet.

To train SPecOnet, execute the code, total2.py.  


After the code has finished running, The computational data are saved as follows:
- 2dforce100sigma5all.csv records Rel. $L^2$ error for $u$, $v$, and $\nabla p$ for $t=k\Delta t$, where $\Delta t=0.01$, $k=1,2,\cdots, 100$.  
- In training/NS2d0.1/uexsigma5all, the $L^2$ norm of reference solutions for each time, $\|u\|_{L^2}^2$, $\|v\|_{L^2}^2$, and $\|\nabla p\|_{L^2}^2$ are saved. 
- In training/NS2d0.1/ubarsigma5all, the $L^2$ norm of inferences for each time, $\| \widehat{u} \|_{L^2}^2$, $\| \widehat{v} \|_{L^2}^2$, and $\|\nabla \widehat{p}\|_{L^2}^2$ are saved. 
- In training/NS2d0.1/pp, reference solution and inferences of $p$ for each time are saved. 

  

## Reference

