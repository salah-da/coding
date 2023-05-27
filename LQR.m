clear all
clc
A=[0 1;...
    0 0];
B=[0;-1];
C=[1 0];
Ai=[A zeros(2,1);C 0];
Bi=[B;0];
Q=[ 100 0 0;...
    0 100 0;...
    0 0  1];
    
R=1;
K=lqr(Ai,Bi,Q,R)
