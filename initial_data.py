from firedrake import *
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


    return (phi);
