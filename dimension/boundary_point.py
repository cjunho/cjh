"This file computes some valuses to design a computaional domain."
"* Function 'bd_x1' computes the value of the left boundary of the domain for SP3."
"* Function 'bd_x2' computes the value of the right boundary of the domain for SP3."
"* Function 'bd_x21' computes the value of the left boundary of the domain for SP2."
"* Function 'bd_x22' computes the value of the right boundary of the domain for SP2."
"* Function 'yast' computes yast such that eta is symmetric about y=yast for SP3."
"  so as to provide the value of the bottom boundary."

import numpy as np

def bd_x1(yy,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu):
    A136 = k6*k6*(k3-k1) + k3*k3*(k1-k6) + k1*k1*(k6-k3)
    A135 = k5*k5*(k3-k1) + k3*k3*(k1-k5) + k1*k1*(k5-k3)
    x1=x_shift-(k6+k5)*(4.5)**(1/6)*ep**.5*yy+((k6**2+k6*k5+k5**2)*(4/3)**(1/3)*ep)*t\
        -(mu/ep)**.5*(2/9)**(1/6)*np.log(A136/A135)/(k6-k5)    
    
    return x1

def bd_x2(yy,x_shift,k1,k2,k3,k4,k5,k6,t,ep,mu):
    A246 = k6*k6*(k4-k2) + k4*k4*(k2-k6) + k2*k2*(k6-k4)
    A146 = k6*k6*(k4-k1) + k4*k4*(k1-k6) + k1*k1*(k6-k4)
    x2=x_shift-(k1+k2)*(4.5)**(1/6)*ep**.5*yy+((k1**2+k1*k2+k2**2)*(4/3)**(1/3)*ep)*t\
        -(mu/ep)**.5*(2/9)**(1/6)*np.log(A246/A146)/(k2-k1)
    
    return x2

def bd_x21(yy,x_shift,k1,k2,k3,k4,t,ep,mu):
    x1=x_shift-k4*(4.5)**(1/6)*ep**.5*yy-(mu/ep)**.5*np.log(2)/(4.5)**(1/6)/k4    
    
    return x1

def bd_x22(yy,x_shift,k1,k2,k3,k4,t,ep,mu):
    x2=x_shift-k1*(4.5)**(1/6)*ep**.5*yy-(mu/ep)**.5*np.log(2)/(4.5)**(1/6)/k1
    
    return x2

def yast(k1,k2,k3,k4,k5,k6):    
    yast=(2/9)**(1/3)*.5*np.log(k5*(k5**2-k4**2)/(k6*(k6**2-k4**2)))/(k6**2-k5**2)
    return yast
    

