"This file defines the initial conditions eta_0(x,y), and Phi_0(x,y)"
"to provide BL_periodic.py the initial conditions"
"• Function 'initial_eta' defines eta_0 for SP3."
"• Function 'initial_phi' defines Phi_0 for SP3."
"• Function 'initial_eta2' defines eta_0 for SP2."
"• Function 'initial_phi2' defines Phi_0 for SP2."

"  Functions initial_phib and initial_phiby are to modify Phi_0 to be periodic at the left boundary, say x=x1, and the right boundary, say x=x2;"
"• Functions initial_phib provides values of Phi_0 on x=x1 or x=x2 for SP3."
"• Functions initial_phiby provides values of (Phi_0)_y on x=x1 or x=x2 for SP3."
"• Functions initial_phib2 provides values of Phi_0 on x=x1 or x=x2 for SP2."
"• Functions initial_phiby2 provides values of (Phi_0)_y on x=x1 or x=x2 for SP2."

from firedrake import *
from boundary_point import *
import numpy as np
def initial_eta(xx,yy,eta,k1,k2,k3,k4,k5,k6,t,ep,mu):   
    K135 = (k1+k3+k5)  
    K235 = (k2+k3+k5)
    K136 = (k1+k3+k6)
    K236 = (k2+k3+k6)
    K145 = (k1+k4+k5)
    K245 = (k2+k4+k5)
    K146 = (k1+k4+k6)
    K246 = (k2+k4+k6)
    
    
    
    KK135 = (k1*k1 + k3*k3 + k5*k5)
    KK235 = (k2*k2 + k3*k3 + k5*k5)
    KK136 = (k1*k1 + k3*k3 + k6*k6)
    KK236 = (k2*k2 + k3*k3 + k6*k6)
    KK145 = (k1*k1 + k4*k4 + k5*k5)
    KK245 = (k2*k2 + k4*k4 + k5*k5)
    KK146 = (k1*k1 + k4*k4 + k6*k6)
    KK246 = (k2*k2 + k4*k4 + k6*k6)
    
    KKK135 = (k1*k1*k1 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK235 = (k2*k2*k2 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK136 = (k1*k1*k1 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK236 = (k2*k2*k2 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK145 = (k1*k1*k1 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK245 = (k2*k2*k2 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK146 = (k1*k1*k1 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK246 = (k2*k2*k2 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    
    t135 = (k1*k1*k1 + k3*k3*k3 + k5*k5*k5)
    t235 = (k2*k2*k2 + k3*k3*k3 + k5*k5*k5)
    t136 = (k1*k1*k1 + k3*k3*k3 + k6*k6*k6)
    t236 = (k2*k2*k2 + k3*k3*k3 + k6*k6*k6)
    t145 = (k1*k1*k1 + k4*k4*k4 + k5*k5*k5)
    t245 = (k2*k2*k2 + k4*k4*k4 + k5*k5*k5)
    t146 = (k1*k1*k1 + k4*k4*k4 + k6*k6*k6)
    t246 = (k2*k2*k2 + k4*k4*k4 + k6*k6*k6)
    
    
    
    T1 = k5*k5*(k3-k1) + k3*k3*(k1-k5) + k1*k1*(k5-k3)
    T2 = k5*k5*(k3-k2) + k3*k3*(k2-k5) + k2*k2*(k5-k3)
    T3 = k6*k6*(k3-k1) + k3*k3*(k1-k6) + k1*k1*(k6-k3)
    T4 = k6*k6*(k3-k2) + k3*k3*(k2-k6) + k2*k2*(k6-k3)
    T5 = k5*k5*(k4-k1) + k4*k4*(k1-k5) + k1*k1*(k5-k4)
    T6 = k5*k5*(k4-k2) + k4*k4*(k2-k5) + k2*k2*(k5-k4)
    T7 = k6*k6*(k4-k1) + k4*k4*(k1-k6) + k1*k1*(k6-k4)
    T8 = k6*k6*(k4-k2) + k4*k4*(k2-k6) + k2*k2*(k6-k4)
    
    
    u2=(( T1*K135*K135*exp( K135*(xx) + KK135*yy-KKK135*t) \
                          +T2*K235*K235*exp( K235*(xx) + KK235*yy-KKK235*t) \
                          +T3*K136*K136*exp( K136*(xx) + KK136*yy-KKK136*t) \
                          +T4*K236*K236*exp( K236*(xx) + KK236*yy-KKK236*t) \
                          +T5*K145*K145*exp( K145*(xx) + KK145*yy-KKK145*t) \
                          +T6*K245*K245*exp( K245*(xx) + KK245*yy-KKK245*t) \
                          +T7*K146*K146*exp( K146*(xx) + KK146*yy-KKK146*t) \
                          +T8*K246*K246*exp( K246*(xx) + KK246*yy-KKK246*t) ))
    
    d1=(T1*exp( K135*(xx) + KK135*yy-KKK135*t) \
                    +T2*exp( K235*(xx) + KK235*yy-KKK235*t) \
                    +T3*exp( K136*(xx) + KK136*yy-KKK136*t) \
                    +T4*exp( K236*(xx) + KK236*yy-KKK236*t) \
                    +T5*exp( K145*(xx) + KK145*yy-KKK145*t) \
                    +T6*exp( K245*(xx) + KK245*yy-KKK245*t) \
                    +T7*exp( K146*(xx) + KK146*yy-KKK146*t) \
                    +T8*exp( K246*(xx) + KK246*yy-KKK246*t))
    
    u1=((T1*K135*exp( K135*(xx) + KK135*yy-KKK135*t)\
                  +T2*K235*exp( K235*(xx) + KK235*yy-KKK235*t ) \
                  +T3*K136*exp( K136*(xx) + KK136*yy-KKK136*t ) \
                  +T4*K236*exp( K236*(xx) + KK236*yy-KKK236*t ) \
                  +T5*K145*exp( K145*(xx) + KK145*yy-KKK145*t ) \
                  +T6*K245*exp( K245*(xx) + KK245*yy-KKK245*t ) \
                  +T7*K146*exp( K146*(xx) + KK146*yy-KKK146*t ) \
                  +T8*K246*exp( K246*(xx) + KK246*yy-KKK246*t )))
    
 
    eta.interpolate(((4/3)**(1/3)*2*(u2/d1-(u1/d1)**2)))
 
    return (eta);

def initial_phi(xx,yy,phi,k1,k2,k3,k4,k5,k6,t,ep,mu):
    K135 = (k1+k3+k5)  
    K235 = (k2+k3+k5)
    K136 = (k1+k3+k6)
    K236 = (k2+k3+k6)
    K145 = (k1+k4+k5)
    K245 = (k2+k4+k5)
    K146 = (k1+k4+k6)
    K246 = (k2+k4+k6)
    
    KK135 = (k1*k1 + k3*k3 + k5*k5)
    KK235 = (k2*k2 + k3*k3 + k5*k5)
    KK136 = (k1*k1 + k3*k3 + k6*k6)
    KK236 = (k2*k2 + k3*k3 + k6*k6)
    KK145 = (k1*k1 + k4*k4 + k5*k5)
    KK245 = (k2*k2 + k4*k4 + k5*k5)
    KK146 = (k1*k1 + k4*k4 + k6*k6)
    KK246 = (k2*k2 + k4*k4 + k6*k6)
    
    KKK135 = (k1*k1*k1 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK235 = (k2*k2*k2 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK136 = (k1*k1*k1 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK236 = (k2*k2*k2 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK145 = (k1*k1*k1 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK245 = (k2*k2*k2 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK146 = (k1*k1*k1 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK246 = (k2*k2*k2 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
   
    T1 = k5*k5*(k3-k1) + k3*k3*(k1-k5) + k1*k1*(k5-k3)
    T2 = k5*k5*(k3-k2) + k3*k3*(k2-k5) + k2*k2*(k5-k3)
    T3 = k6*k6*(k3-k1) + k3*k3*(k1-k6) + k1*k1*(k6-k3)
    T4 = k6*k6*(k3-k2) + k3*k3*(k2-k6) + k2*k2*(k6-k3)
    T5 = k5*k5*(k4-k1) + k4*k4*(k1-k5) + k1*k1*(k5-k4)
    T6 = k5*k5*(k4-k2) + k4*k4*(k2-k5) + k2*k2*(k5-k4)
    T7 = k6*k6*(k4-k1) + k4*k4*(k1-k6) + k1*k1*(k6-k4)
    T8 = k6*k6*(k4-k2) + k4*k4*(k2-k6) + k2*k2*(k6-k4)
    
    
    d1=(T1*exp( K135*(xx) + KK135*yy-KKK135*t) \
                    +T2*exp( K235*(xx) + KK235*yy-KKK235*t) \
                    +T3*exp( K136*(xx) + KK136*yy-KKK136*t) \
                    +T4*exp( K236*(xx) + KK236*yy-KKK236*t) \
                    +T5*exp( K145*(xx) + KK145*yy-KKK145*t) \
                    +T6*exp( K245*(xx) + KK245*yy-KKK245*t) \
                    +T7*exp( K146*(xx) + KK146*yy-KKK146*t) \
                    +T8*exp( K246*(xx) + KK246*yy-KKK246*t))
    
    u1=((T1*K135*exp( K135*(xx) + KK135*yy-KKK135*t)\
                  +T2*K235*exp( K235*(xx) + KK235*yy-KKK235*t ) \
                  +T3*K136*exp( K136*(xx) + KK136*yy-KKK136*t ) \
                  +T4*K236*exp( K236*(xx) + KK236*yy-KKK236*t ) \
                  +T5*K145*exp( K145*(xx) + KK145*yy-KKK145*t ) \
                  +T6*K245*exp( K245*(xx) + KK245*yy-KKK245*t ) \
                  +T7*K146*exp( K146*(xx) + KK146*yy-KKK146*t ) \
                  +T8*K246*exp( K246*(xx) + KK246*yy-KKK246*t )))
        
    
    
    phi.interpolate((32/81)**(1/6)*(mu/ep)**(.5)*(2*u1/d1+1))


    return (phi);

def initial_phib(xx,yy,k1,k2,k3,k4,k5,k6,t,ep,mu):

    K135 = (k1+k3+k5)  
    K235 = (k2+k3+k5)
    K136 = (k1+k3+k6)
    K236 = (k2+k3+k6)
    K145 = (k1+k4+k5)
    K245 = (k2+k4+k5)
    K146 = (k1+k4+k6)
    K246 = (k2+k4+k6)
    
    KK135 = (k1*k1 + k3*k3 + k5*k5)
    KK235 = (k2*k2 + k3*k3 + k5*k5)
    KK136 = (k1*k1 + k3*k3 + k6*k6)
    KK236 = (k2*k2 + k3*k3 + k6*k6)
    KK145 = (k1*k1 + k4*k4 + k5*k5)
    KK245 = (k2*k2 + k4*k4 + k5*k5)
    KK146 = (k1*k1 + k4*k4 + k6*k6)
    KK246 = (k2*k2 + k4*k4 + k6*k6)
    
    KKK135 = (k1*k1*k1 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK235 = (k2*k2*k2 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK136 = (k1*k1*k1 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK236 = (k2*k2*k2 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK145 = (k1*k1*k1 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK245 = (k2*k2*k2 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK146 = (k1*k1*k1 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK246 = (k2*k2*k2 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    

    T1 = k5*k5*(k3-k1) + k3*k3*(k1-k5) + k1*k1*(k5-k3)
    T2 = k5*k5*(k3-k2) + k3*k3*(k2-k5) + k2*k2*(k5-k3)
    T3 = k6*k6*(k3-k1) + k3*k3*(k1-k6) + k1*k1*(k6-k3)
    T4 = k6*k6*(k3-k2) + k3*k3*(k2-k6) + k2*k2*(k6-k3)
    T5 = k5*k5*(k4-k1) + k4*k4*(k1-k5) + k1*k1*(k5-k4)
    T6 = k5*k5*(k4-k2) + k4*k4*(k2-k5) + k2*k2*(k5-k4)
    T7 = k6*k6*(k4-k1) + k4*k4*(k1-k6) + k1*k1*(k6-k4)
    T8 = k6*k6*(k4-k2) + k4*k4*(k2-k6) + k2*k2*(k6-k4)
    
    
    d1=(T1*exp( K135*(xx) + KK135*yy-KKK135*t) \
                    +T2*exp( K235*(xx) + KK235*yy-KKK235*t) \
                    +T3*exp( K136*(xx) + KK136*yy-KKK136*t) \
                    +T4*exp( K236*(xx) + KK236*yy-KKK236*t) \
                    +T5*exp( K145*(xx) + KK145*yy-KKK145*t) \
                    +T6*exp( K245*(xx) + KK245*yy-KKK245*t) \
                    +T7*exp( K146*(xx) + KK146*yy-KKK146*t) \
                    +T8*exp( K246*(xx) + KK246*yy-KKK246*t))
    
    u1=((T1*K135*exp( K135*(xx) + KK135*yy-KKK135*t)\
                  +T2*K235*exp( K235*(xx) + KK235*yy-KKK235*t ) \
                  +T3*K136*exp( K136*(xx) + KK136*yy-KKK136*t ) \
                  +T4*K236*exp( K236*(xx) + KK236*yy-KKK236*t ) \
                  +T5*K145*exp( K145*(xx) + KK145*yy-KKK145*t ) \
                  +T6*K245*exp( K245*(xx) + KK245*yy-KKK245*t ) \
                  +T7*K146*exp( K146*(xx) + KK146*yy-KKK146*t ) \
                  +T8*K246*exp( K246*(xx) + KK246*yy-KKK246*t )))    
    phi=((32/81)**(1/6)*(mu/ep)**(.5)*(2*u1/d1+1))


    return (phi);

def initial_phiby(xx,yy,k1,k2,k3,k4,k5,k6,t,ep,mu):

    K135 = (k1+k3+k5)  
    K235 = (k2+k3+k5)
    K136 = (k1+k3+k6)
    K236 = (k2+k3+k6)
    K145 = (k1+k4+k5)
    K245 = (k2+k4+k5)
    K146 = (k1+k4+k6)
    K246 = (k2+k4+k6)
    
    KK135 = (k1*k1 + k3*k3 + k5*k5)
    KK235 = (k2*k2 + k3*k3 + k5*k5)
    KK136 = (k1*k1 + k3*k3 + k6*k6)
    KK236 = (k2*k2 + k3*k3 + k6*k6)
    KK145 = (k1*k1 + k4*k4 + k5*k5)
    KK245 = (k2*k2 + k4*k4 + k5*k5)
    KK146 = (k1*k1 + k4*k4 + k6*k6)
    KK246 = (k2*k2 + k4*k4 + k6*k6)
    
    KKK135 = (k1*k1*k1 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK235 = (k2*k2*k2 + k3*k3*k3 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK136 = (k1*k1*k1 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK236 = (k2*k2*k2 + k3*k3*k3 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK145 = (k1*k1*k1 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK245 = (k2*k2*k2 + k4*k4*k4 + k5*k5*k5)*ep*(2*ep/mu)**(1/2)
    KKK146 = (k1*k1*k1 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    KKK246 = (k2*k2*k2 + k4*k4*k4 + k6*k6*k6)*ep*(2*ep/mu)**(1/2)
    

    T1 = k5*k5*(k3-k1) + k3*k3*(k1-k5) + k1*k1*(k5-k3)
    T2 = k5*k5*(k3-k2) + k3*k3*(k2-k5) + k2*k2*(k5-k3)
    T3 = k6*k6*(k3-k1) + k3*k3*(k1-k6) + k1*k1*(k6-k3)
    T4 = k6*k6*(k3-k2) + k3*k3*(k2-k6) + k2*k2*(k6-k3)
    T5 = k5*k5*(k4-k1) + k4*k4*(k1-k5) + k1*k1*(k5-k4)
    T6 = k5*k5*(k4-k2) + k4*k4*(k2-k5) + k2*k2*(k5-k4)
    T7 = k6*k6*(k4-k1) + k4*k4*(k1-k6) + k1*k1*(k6-k4)
    T8 = k6*k6*(k4-k2) + k4*k4*(k2-k6) + k2*k2*(k6-k4)
    
    
    d1=(T1*exp( K135*(xx) + KK135*yy-KKK135*t) \
                    +T2*exp( K235*(xx) + KK235*yy-KKK235*t) \
                    +T3*exp( K136*(xx) + KK136*yy-KKK136*t) \
                    +T4*exp( K236*(xx) + KK236*yy-KKK236*t) \
                    +T5*exp( K145*(xx) + KK145*yy-KKK145*t) \
                    +T6*exp( K245*(xx) + KK245*yy-KKK245*t) \
                    +T7*exp( K146*(xx) + KK146*yy-KKK146*t) \
                    +T8*exp( K246*(xx) + KK246*yy-KKK246*t))
    
    u1=((T1*K135*exp( K135*(xx) + KK135*yy-KKK135*t)\
                  +T2*K235*exp( K235*(xx) + KK235*yy-KKK235*t ) \
                  +T3*K136*exp( K136*(xx) + KK136*yy-KKK136*t ) \
                  +T4*K236*exp( K236*(xx) + KK236*yy-KKK236*t ) \
                  +T5*K145*exp( K145*(xx) + KK145*yy-KKK145*t ) \
                  +T6*K245*exp( K245*(xx) + KK245*yy-KKK245*t ) \
                  +T7*K146*exp( K146*(xx) + KK146*yy-KKK146*t ) \
                  +T8*K246*exp( K246*(xx) + KK246*yy-KKK246*t )))
    uy=((T1*KK135*exp( K135*(xx) + KK135*yy-KKK135*t)\
                  +T2*KK235*exp( K235*(xx) + KK235*yy-KKK235*t ) \
                  +T3*KK136*exp( K136*(xx) + KK136*yy-KKK136*t ) \
                  +T4*KK236*exp( K236*(xx) + KK236*yy-KKK236*t ) \
                  +T5*KK145*exp( K145*(xx) + KK145*yy-KKK145*t ) \
                  +T6*KK245*exp( K245*(xx) + KK245*yy-KKK245*t ) \
                  +T7*KK146*exp( K146*(xx) + KK146*yy-KKK146*t ) \
                  +T8*KK246*exp( K246*(xx) + KK246*yy-KKK246*t )))
    uxy=((T1*K135*KK135*exp( K135*(xx) + KK135*yy-KKK135*t)\
                  +T2*K235*KK235*exp( K235*(xx) + KK235*yy-KKK235*t ) \
                  +T3*K136*KK136*exp( K136*(xx) + KK136*yy-KKK136*t ) \
                  +T4*K236*KK236*exp( K236*(xx) + KK236*yy-KKK236*t ) \
                  +T5*K145*KK145*exp( K145*(xx) + KK145*yy-KKK145*t ) \
                  +T6*K245*KK245*exp( K245*(xx) + KK245*yy-KKK245*t ) \
                  +T7*K146*KK146*exp( K146*(xx) + KK146*yy-KKK146*t ) \
                  +T8*K246*KK246*exp( K246*(xx) + KK246*yy-KKK246*t )))
    phi=((8)**(1/6)*(ep)**(.5)*(2*(uxy/d1-(u1/d1)*(uy/d1))))


    return (phi)

def initial_eta2(xx,yy,eta,k1,k2,k3,k4,t,ep,mu):
    
    K13 = (k1+k3)  
    K23 = (k2+k3)    
    K14 = (k1+k4)
    K24 = (k2+k4)

    KK13 = (k1*k1 + k3*k3)
    KK23 = (k2*k2 + k3*k3)   
    KK14 = (k1*k1 + k4*k4)
    KK24 = (k2*k2 + k4*k4)
    
    KKK13 = (k1*k1*k1 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK23 = (k2*k2*k2 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK14 = (k1*k1*k1 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)
    KKK24 = (k2*k2*k2 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)   
    
    
    T13 = k3-k1
    T23 = k3-k2
    T14 = k4-k1
    T24 = k4-k2
    
    
    d1=(T13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*exp( K24*(xx) + KK24*yy-KKK24*t))
    
    u1=(T13*K13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*K23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*K14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*K24*exp( K24*(xx) + KK24*yy-KKK24*t))
    u2=(T13*K13*K13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*K23*K23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*K14*K14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*K24*K24*exp( K24*(xx) + KK24*yy-KKK24*t))

    eta.interpolate(((4/3)**(1/3)*2*(u2/d1-(u1/d1)**2)))

    return (eta)

def initial_phi2(xx,yy,phi,k1,k2,k3,k4,t,ep,mu):
    
    K13 = (k1+k3)  
    K23 = (k2+k3)    
    K14 = (k1+k4)
    K24 = (k2+k4)    
    
    
    KK13 = (k1*k1 + k3*k3)
    KK23 = (k2*k2 + k3*k3)   
    KK14 = (k1*k1 + k4*k4)
    KK24 = (k2*k2 + k4*k4)
    
    KKK13 = (k1*k1*k1 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK23 = (k2*k2*k2 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK14 = (k1*k1*k1 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)
    KKK24 = (k2*k2*k2 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)   
    
    
    T13 = k3-k1
    T23 = k3-k2
    T14 = k4-k1
    T24 = k4-k2
    
    
    d1=(T13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*exp( K24*(xx) + KK24*yy-KKK24*t))
    
    u1=(T13*K13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*K23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*K14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*K24*exp( K24*(xx) + KK24*yy-KKK24*t))
    
    phi.interpolate((32/81)**(1/6)*(mu/ep)**(.5)*(2*u1/d1+1))


    return (phi)

def initial_phib2(xx,yy,k1,k2,k3,k4,t,ep,mu):
    
    K13 = (k1+k3)  
    K23 = (k2+k3)    
    K14 = (k1+k4)
    K24 = (k2+k4)    
    
    
    KK13 = (k1*k1 + k3*k3)
    KK23 = (k2*k2 + k3*k3)   
    KK14 = (k1*k1 + k4*k4)
    KK24 = (k2*k2 + k4*k4)
    
    KKK13 = (k1*k1*k1 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK23 = (k2*k2*k2 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK14 = (k1*k1*k1 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)
    KKK24 = (k2*k2*k2 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)   
    
    
    T13 = k3-k1
    T23 = k3-k2
    T14 = k4-k1
    T24 = k4-k2
    
    
    d1=(T13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*exp( K24*(xx) + KK24*yy-KKK24*t))
    
    u1=(T13*K13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*K23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*K14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*K24*exp( K24*(xx) + KK24*yy-KKK24*t))
    uxy=(T13*KK13*K13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*KK23*K23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*KK14*K14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*KK24*K24*exp( K24*(xx) + KK24*yy-KKK24*t)) 
    uy=(T13*KK13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*KK23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*KK14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*KK24*exp( K24*(xx) + KK24*yy-KKK24*t)) 
    phi=((32/81)**(1/6)*(mu/ep)**(.5)*(2*u1/d1+1))


    return (phi)

def initial_phiby2(xx,yy,k1,k2,k3,k4,t,ep,mu):
    
    K13 = (k1+k3)  
    K23 = (k2+k3)    
    K14 = (k1+k4)
    K24 = (k2+k4)    
    
    # (abs(coords[1]+20)-20)
    KK13 = (k1*k1 + k3*k3)
    KK23 = (k2*k2 + k3*k3)   
    KK14 = (k1*k1 + k4*k4)
    KK24 = (k2*k2 + k4*k4)
    
    KKK13 = (k1*k1*k1 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK23 = (k2*k2*k2 + k3*k3*k3)*ep*(2*ep/mu)**(1/2)
    KKK14 = (k1*k1*k1 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)
    KKK24 = (k2*k2*k2 + k4*k4*k4)*ep*(2*ep/mu)**(1/2)   
    
    
    T13 = k3-k1
    T23 = k3-k2
    T14 = k4-k1
    T24 = k4-k2
    
    
    d1=(T13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*exp( K24*(xx) + KK24*yy-KKK24*t))
    
    u1=(T13*K13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*K23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*K14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*K24*exp( K24*(xx) + KK24*yy-KKK24*t))
    uxy=(T13*KK13*K13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*KK23*K23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*KK14*K14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*KK24*K24*exp( K24*(xx) + KK24*yy-KKK24*t)) 
    uy=(T13*KK13*exp( K13*(xx) + KK13*yy-KKK13*t) \
                    +T23*KK23*exp( K23*(xx) + KK23*yy-KKK23*t)\
                    +T14*KK14*exp( K14*(xx) + KK14*yy-KKK14*t) \
                    +T24*KK24*exp( K24*(xx) + KK24*yy-KKK24*t)) 
    phi=((8)**(1/6)*(ep)**(.5)*(2*(uxy/d1-(u1/d1)*(uy/d1))))


    return (phi)
