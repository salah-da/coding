classdef Drone < handle
    
%% MEMBERS    
    properties
        g
        t
        dt
        tf
        
        m
        l
        I
        kd
        kt
        x                                                                  %(X, Y, Z, dX, dY, dZ, phi, theta, psi, p, q, r)
        r                                                                  %(X, Y, Z)
        dr                                                                 %(dX, dY, dZ)
        euler                                                              %(phi, theta, psi)
        w                                                                  %(p, q, r)
        RPM
        dx
        du
        u
        T
        M
    end
    
    properties
        
        e
        uk
        phi_des
        phi_err
        phi_err_prev
        phi_err_sum
        
        theta_des
        theta_err
        theta_err_prev
        theta_err_sum
        
        psi_des
        psi_err
        psi_err_prev
        psi_err_sum
        
        z_des
        z_err
        z_err_prev
        z_err_sum
    end
    
    properties
        KP_phi
        KI_phi
        KD_phi
        
        KP_theta
        KI_theta
        KD_theta
        
        KP_psi
        KI_psi
        KD_psi
        
        KP_z
        KI_z
        KD_z
    end
    
    
%% METHODS
    methods
    %% CONSTRUCTOR
        function obj = Drone(params, initStates, initInputs, gains, simTime)
            obj.g = 9.81;
            obj.t = 0.0;
            obj.dt = 0.09;
            obj.tf = simTime;
            
            obj.m = params('mass');
            obj.l = params('armLength');
            obj.I = [params('Ixx'),0,0 ; 0,params('Iyy'),0; 0,0,params('Izz')];
            
            
            obj.x = initStates;
            obj.r = obj.x(1:3);
            obj.dr = obj.x(4:6);
            obj.euler = obj.x(7:9);
            obj.w = obj.x(10:12);
            
            obj.dx = zeros(12,1);
            obj.du=zeros(4,1);
            obj.u = initInputs;
            obj.T = obj.u(1);
            obj.M = obj.u(2:4);
            
            obj.e(1)=0;
            obj.e(2)=0;
            obj.e(3)=0;
            obj.uk(1)=0;
            obj.uk(2)=0;
         
			obj.phi_err = 0.0;
            obj.phi_err_prev = 0.0;
            obj.phi_err_sum = 0.0;
            obj.theta_err = 0.0;
            obj.theta_err_prev = 0.0;
            obj.theta_err_sum = 0.0;
            obj.psi_err = 0.0;
            obj.psi_err_prev = 0.0;
            obj.psi_err_sum = 0.0;
            
            obj.z_err = 0.0;
            obj.z_err_prev = 0.0;
            obj.z_err_sum = 0.0;
          
            obj.KP_phi=gains('P_phi');
            obj.KI_phi=gains('I_phi');
            obj.KD_phi=gains('D_phi');
            
            obj.KP_theta=gains('P_theta');
            obj.KI_theta=gains('I_theta');
            obj.KD_theta=gains('D_theta');
            
            obj.KP_psi=gains('P_psi');
            obj.KI_psi=gains('I_psi');
            obj.KD_psi=gains('D_psi');
            
            obj.KP_z = gains('P_z');
            obj.KI_z = gains('I_z');
            obj.KD_z = gains('D_z');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
      
    %% RETURNS DRONE STATE
        function [state,RPM,u] = GetState(obj)
            state = obj.x;
            RPM=obj.RPM;
            u=obj.u(1);
        end
        
    %% STATE SPACE (DIFFERENTIAL) EQUATIONS
        function obj = EvalEOM(obj)
            R = RPY2Rot(obj.euler);            
                                                             
           
            % Translational Motions
            obj.dx(1:3) = obj.dr;
           obj.dx(4:6) = 1 / obj.m *(R*[0;0;obj.T])-[0;0;obj.g];
          
            % Rotational Motions
            phi = obj.euler(1); theta = obj.euler(2);
            Rt = [ 1 sin(phi)*tan(theta) cos(phi)*tan(theta);
    0 cos(phi) -sin(phi);0 sin(phi)*sec(theta) cos(phi)*sec(theta)];
            obj.dx(7:9) = Rt*obj.w;
                       
            obj.dx(10:12) =(obj.I\(obj.M - cross(obj.w, obj.I * obj.w)));

        end

    %% PREDICT NEXT DRONE STATE
        function obj = UpdateState(obj)
            obj.t = obj.t + obj.dt;
           
            % Find(update) the next state of obj.X
           
             
            obj.EvalEOM();
            obj.x = obj.x + obj.dx.*obj.dt;
            
            
            obj.r = obj.x(1:3);
            obj.dr = obj.x(4:6);
            obj.euler = obj.x(7:9);
            obj.w = obj.x(10:12);
            
        end
    %% CONTROLLER
    
        function obj = AttitudeCtrl(obj, refSig)
			obj.phi_des = refSig(1);
			obj.theta_des =  refSig(2);
			obj.psi_des = refSig(3);
			obj.z_des = refSig(4);
 			 obj.phi_err = obj.phi_des - obj.euler(1);
            obj.theta_err = obj.theta_des - obj.euler(2);
            obj.psi_err = obj.psi_des - obj.euler(3);
            obj.z_err = obj.z_des - obj.r(3);
            %obj.u(1) = obj.m * obj.g;
            obj.u(1) = (obj.KP_z * obj.z_err + ...
                        obj.KI_z * (obj.z_err_sum) + ...
                        obj.KD_z * (0-obj.dr(3)));
                    %obj.z_err - obj.z_err_prev)/obj.dt
            obj.z_err_prev = obj.z_err;
            obj.z_err_sum = obj.z_err_sum + obj.dt*obj.z_err;
             obj.u(2)=0;
             obj.u(3)=0;
            obj.u(4)=0;
            
            a=1/(cos(obj.euler(1))*cos(obj.euler(2)));
            obj.T = a*obj.m*(obj.u(1)+obj.g);
            obj.M = obj.u(2:4);
            obj.RPM=motor([obj.T ;obj.M]);
        end
        function obj=pid_d(obj,refSig)
            obj.phi_des = refSig(1);
			obj.theta_des =  refSig(2);
			obj.psi_des = refSig(3);
			obj.z_des = refSig(4);
 			 obj.phi_err = obj.phi_des - obj.euler(1);
            obj.theta_err = obj.theta_des - obj.euler(2);
            obj.psi_err = obj.psi_des - obj.euler(3);
              obj.z_err = obj.z_des - obj.r(3);
            obj.e(1)=obj.z_err;
            
           
           % a=1/(cos(obj.euler(1))*cos(obj.euler(2)));a*obj.m*(obj.g+
            obj.u(1) = (pid_dis( [obj.KP_z;obj.KI_z;obj.KD_z],obj.dt,100,obj.e,obj.uk));
            obj.e(3)=obj.e(2);
            obj.e(2)=obj.e(1);
            obj.uk(2)=obj.uk(1);
            obj.uk(1)=obj.u(1);
            obj.u(2)=0;
            obj.u(3)=0;
            obj.u(4)=0;
            obj.T = obj.u(1);
            obj.M = obj.u(2:4);
            obj.RPM=motor([obj.T ;obj.M]);
        end
    end
end


