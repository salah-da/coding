function  omega= motor( u )
kt=1.75*10^-5;
kd=2.74*10^-7;
l=0.255;
a=sqrt(2)/2;
m=[kt kt kt kt;...
    kt*l*a a*kt*l -a*kt*l -a*kt*l;...
    -a*kt*l a*kt*l a*kt*l -a*kt*l;...
    kd -kd kd -kd];
d=inv(m);
omega=sqrt(d*u)*9.5492968;

end

