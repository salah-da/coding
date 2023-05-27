%pso parameters
itr=100; %number of iteration 
N=30; %number of population
c=2;
wmax=0.7;
wmin=0.2;
var=3;
%search space
lb=0;
ub=50;
%optimization steps
c_cf=0;
%ref
dt=0.09;
R2D = 180/pi;
D2R = pi/180;
%% INIT. PARAMS.
drone1_params = containers.Map({'mass','armLength','Ixx','Iyy','Izz'},...
    {0.920, 0.255, 0.0108, 0.0108, 0.0212});

drone1_initStates = [0, 0, 0, ...                                              % x, y, z
    0, 0, 0, ...                                                                % dx, dy, dz
    0, 0, 0, ...                                                                % phi, theta, psi
    0, 0, 0]';                                                                  % p, q, r

drone1_initInputs = [3*3, ...                                                     % ThrottleCMD
    0, 0, 0]';                                                                  % R, P, Y CMD

drone1_body = [ 0.255,      0,     0, 1; ...
                    0, -0.255,     0, 1; ...
               -0.255,      0,     0, 1; ...
                    0,  0.255,     0, 1; ...
                    0,      0,     0, 1; ...
                    0,      0, -0.15, 1]';
			  
drone1_gains = containers.Map(...
	{'P_phi','I_phi','D_phi',...
	'P_theta','I_theta','D_theta',...
	'P_psi','I_psi','D_psi',...
	'P_z','I_z','D_z'},...
    {0.0, 0.0, 0.0,...
	0.0, 0.0, 0.0,...
	0.0, 0.0, 0.0,...
	3.0, 4, 2.0});

simulationTime = 10;                                                          % [sec]


 
commandSig(1) = 20.0*D2R; %phi
commandSig(2) = 00.0*D2R; %theta
commandSig(3) = 00.0*D2R; %psi
commandSig(4) = 3; %z

%initialisation
for m=1:N
    for n=1:var
        v(m,n)=0;
        x(m,n)=lb+rand*(ub-lb);
        xp(m,n)=x(m,n);
    end
    kp=x(m,1);
    ki=x(m,2);
    kd=x(m,3);
    %% BIRTH OF A DRONE
    drone1_gains('P_z')=kp;
    drone1_gains('I_z')=ki;
    drone1_gains('D_z')=kd;
drone1 = Drone(drone1_params, drone1_initStates, drone1_initInputs, drone1_gains, simulationTime);
drone1_state = drone1.GetState();
    %model simulation
    for f=1:simulationTime/dt
        drone1.getcommand(commandSig);
    drone1.UpdateState();
	drone1_state = drone1.GetState();
    y(f)=drone1_state(3);
    end
    %objective function
    ffb(m)=0;
    for i1=1:simulationTime/dt
        ffb(m)=ffb(m)+dt*(abs((y(i1)-commandSig(4)))*(i1/dt));
    end
end
[fg,gbest]=min(ffb);
xg=x(gbest,:);
for i=1:itr
    w=wmax-((wmax-wmin)*(i/itr));
    for m=i:N
        
        for n=1:var
            v(m,n)=(w*v(m,n))+(c*rand*(xp(m,n)-x(m,n)))+(c*rand*(xg(n)-x(m,n)));
            x(m,n)=x(m,n)+v(m,n);
        end
        for n=1:var
            if x(m,n)<lb;x(m,n)=lb; end
            if x(m,n)>ub;x(m,n)=ub; end
        end
     kp=x(m,1);
    ki=x(m,2);
    kd=x(m,3);
    
    %we repeat the simulation 
    %% BIRTH OF A DRONE
    drone1_gains('P_z')=kp;
    drone1_gains('I_z')=ki;
    drone1_gains('D_z')=kd;
drone1 = Drone(drone1_params, drone1_initStates, drone1_initInputs, drone1_gains, simulationTime);
drone1_state = drone1.GetState();
     %model simulation
    for f=1:simulationTime/dt
        drone1.getcommand(commandSig);
    drone1.UpdateState();
	drone1_state = drone1.GetState();
    y(f)=drone1_state(3);
    end
    %objective function
    ff(m)=0;
    for i1=1:simulationTime/dt
        ff(m)=ff(m)+dt*(abs(y(i1)-commandSig(4))*(i1/dt));
    end
    %compare local
    if ff(m)<ffb(m)
        ffb(m)=ff(m);
        xp(m,:)=x(m,:);
    end
    end
    [Bfg,location]=min(ffb);
    %compare global
    if Bfg<fg
        fg=Bfg;
        xg=xp(location,:);
    end
    %draw the cost function 
%     c_cf=c_cf+1;
%     best_of_pso(c_ccf)=fg;
%     t=1:c_cf;
%     plot(t,best_of_pso)
%     hold on
end
        
