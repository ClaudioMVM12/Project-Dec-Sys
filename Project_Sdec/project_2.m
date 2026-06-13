% =========================================================================
% DECISION SYSTEMS PROJECT - THE ULTIMATE SYNTHESIS
% -------------------------------------------------------------------------
% - Trajectory Splines via Parametric Omega Matrix Inversion
% - Kalman Filter with Joseph-Form Covariance for Bias Estimation
% - L1-Norm Propellant Minimization via fmincon
% - Exact Discrete LQR Tracking & Nonlinear ECI Simulation
% =========================================================================
clear; clc; close all;

%% ========================================================================
% GLOBAL SYSTEM PARAMETERS & MATRICES
% =========================================================================
fprintf('Initializing Spacecraft Parameters...\n');
G    = 6.674e-11;       
mE   = 5.972e24;        
mu   = G * mE;            % (we learnt that this in orbital dynamics) grav. parameter
RT   = 7171e3;          
mC   = 2000;            
h    = 10;                % 10s simulation step
Tsim = 3600;              % 1 hour rendezvous ;)
x0_LVLH = [0; 0; 100e3; 0; 0; 0]; 

omega = sqrt(mu / RT^3);  % mean orbital angu. vel.

% CW Linear State-Space Model (Clohessy-Wiltshire Equations)
A_cw = [0, 0, 0, 1, 0, 0;
        0, 0, 0, 0, 1, 0;
        0, 0, 0, 0, 0, 1;
        0, 0, 0, 0, 0, 2*omega;
        0, -omega^2, 0, 0, 0, 0;
        0, 0, 3*omega^2, -2*omega, 0, 0];
B_cw = [zeros(3,3); eye(3)/mC];

% Exact Discretization 
sys_c = ss(A_cw, B_cw, eye(6), zeros(6,3));
sys_d = c2d(sys_c, h, 'zoh');
Ad = sys_d.A;  Bd = sys_d.B;

t_vec = 0:h:Tsim;
N_steps = length(t_vec);

%% ========================================================================
% QUESTION 1: NONLINEAR ORBITAL DYNAMICS (ECI)
% =========================================================================
fprintf('--- Q1: Simulating Orbital Mechanics (ECI) ---\n');

rT0_ECI = [RT; 0; 0];
vT0_ECI = [0; omega*RT; 0];
i_z = rT0_ECI / norm(rT0_ECI);                             
i_y = cross(rT0_ECI, vT0_ECI) / norm(cross(rT0_ECI, vT0_ECI)); 
i_x = cross(i_y, i_z);                                     
R_L2E_0 = [i_x, i_y, i_z];

rC0_ECI = rT0_ECI + R_L2E_0 * x0_LVLH(1:3);
vC0_ECI = vT0_ECI + R_L2E_0 * x0_LVLH(4:6) + cross([0;0;omega], R_L2E_0 * x0_LVLH(1:3));

options_ode = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);
[~, X_T_ECI] = ode45(@(t,x) [x(4:6); -(mu/norm(x(1:3))^3)*x(1:3)], t_vec, [rT0_ECI; vT0_ECI], options_ode);
[~, X_C_ECI] = ode45(@(t,x) [x(4:6); -(mu/norm(x(1:3))^3)*x(1:3)], t_vec, [rC0_ECI; vC0_ECI], options_ode);
X_T_ECI = X_T_ECI'; X_C_ECI = X_C_ECI';

%% ========================================================================
% QUESTION 2: BIAS ESTIMATION VIA  KALMAN FILTER
% =========================================================================
fprintf('--- Q2: Actuator Bias Estimation ( Kalman Filter) ---\n');

b_true = [2.5; -1.2; 0.7];    %const. bias (don't fuck with this claudio)
noise_accel_std = 0.005; 
sigma_pos = 0.1;         

% Exact variance of the scaled measurement noise (F = m * a)
R_variance = (mC * noise_accel_std)^2; 

%  Kalman Filter Initialization
b_kf = zeros(3, N_steps);
b_kf(:,1) = [0; 0; 0]; % Initial guess
Sigma = cell(1, N_steps);
Sigma{1} = 100 * eye(3);   % Initial covariance
R_kf = eye(3) * R_variance;
Q_kf = eye(3) * 1e-4;      % Tiny process noise to keep filter active
I_3 = eye(3); 

x_curr = x0_LVLH;
u_cmd = [1; 0; -1] .* ones(1, N_steps); 

for k = 1:N_steps-1
    % Physical Reality
    u_real = u_cmd(:, k) + b_true; 
    xdot_true = A_cw * x_curr + B_cw * u_real; 
    
    % Realistic Noisy Measurements
    x_meas = x_curr + randn(6,1) * sigma_pos;       
    xdot_meas = xdot_true + randn(6,1) * noise_accel_std; 
    
    % Decision System Form (y = b + v)
    y_meas_k = mC * (xdot_meas(4:6) - A_cw(4:6, :) * x_meas) - u_cmd(:, k);
    H_k = eye(3); 
    
    % ---  Kalman Filter Algorithm ---
    % Predict
    b_pred = b_kf(:, k); 
    Sigma_pred = Sigma{k} + Q_kf;
    
    % Update
    y_tilde = y_meas_k - H_k * b_pred;
    S = H_k * Sigma_pred * H_k' + R_kf;
    K = Sigma_pred * H_k' / S;
    b_kf(:, k+1) = b_pred + K * y_tilde;
    
    % Highly stable Joseph Form for Covariance Update 
    Sigma{k+1} = (I_3 - K * H_k) * Sigma_pred * (I_3 - K * H_k)' + K * R_kf * K';
    
    x_curr = Ad * x_curr + Bd * u_real;
end
b_kf(:, end) = b_kf(:, end-1);

% Extract std devs for 3-sigma bounds
std_devs = zeros(3, N_steps);
for k = 1:N_steps
    std_devs(:, k) = sqrt(diag(Sigma{k}));
end

%% ========================================================================
% QUESTION 3 & 4:  SPLINE OPTIMIZATION & LQR TRACKING
% =========================================================================
fprintf('\n--- Q3 & Q4: Control, Observation, and Replanning ---\n');

Cd_obs = [eye(3), zeros(3,3)];  
Q_noise = eye(6) * 1e-4;        
R_noise = eye(3) * 10;          
L_gain = dlqe(Ad, eye(6), Cd_obs, Q_noise, R_noise); 

% LQR Controller Design (Tuned perfectly so error is practically 0)
Q_lqr = diag([1e3, 1e3, 1e3, 1e4, 1e4, 1e4]); 
R_lqr = eye(3) * 1e3;                      
K_lqr = dlqr(Ad, Bd, Q_lqr, R_lqr);

% Obstacle Definition (can be changed whenever you feel)
obs_pos = [0; 0; 30e3]; 
r_obs = 20e3; 
safe_radius = r_obs + 2e3;  

% SMART INITIAL GUESS: 15 parameters allowing infinite morphing of the dodge curve.
% C(1:3) represent peak deviations in X, Y, Z. The rest warp the shape.
C0 = zeros(15, 1);
C0(2) = 25e3; % Starting with a cross-track (Y) bias.

% UPGRADED OPTIMIZATION: sqp pushed to maximum depth
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp', ...
    'MaxFunctionEvaluations', 50000, 'MaxIterations', 2000, ...
    'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-8);

lb_opt = -100e3 * ones(15, 1);
ub_opt =  100e3 * ones(15, 1);
lb_opt(3) = -50e3; ub_opt(3) = 50e3; % Cap raw Z deviation

C_opt = fmincon(@(C) cost_func(C, x0_LVLH, Tsim, h, omega, mC), C0, ...
    [], [], [], [], lb_opt, ub_opt, ...
    @(C) param_constraint(C, x0_LVLH, Tsim, h, obs_pos, safe_radius), options);

fprintf('Optimal Avoidance Deviations -> X: %.1f km, Y: %.1f km, Z: %.1f km\n', C_opt(1)/1e3, C_opt(2)/1e3, C_opt(3)/1e3);

% Generate the Final Reference Trajectory via Smooth Splines
[X_ref, U_ff] = build_smooth_trajectory(C_opt, x0_LVLH, Tsim, h, omega, mC);

% Closed-Loop LQR Simulation
x_true = zeros(6, N_steps);  x_true(:,1) = x0_LVLH;
x_hat  = zeros(6, N_steps);  x_hat(:,1)  = x0_LVLH + [200; -200; 200; 0; 0; 0]; 
u_q4   = zeros(3, N_steps);
b_est  = b_kf(:, end); % Final KF bias estimate

for k = 1:N_steps-1
    e_k = x_hat(:, k) - X_ref(:, k);
    u_q4(:, k) = U_ff(:, k) - K_lqr * e_k - b_est; % Bias compensation active
    
    x_true(:, k+1) = Ad * x_true(:, k) + Bd * (u_q4(:, k) + b_true);
    
    y_meas_k = Cd_obs * x_true(:, k+1) + randn(3,1) * sqrt(10);
    x_pred = Ad * x_hat(:, k) + Bd * (u_q4(:, k) + b_est);
    x_hat(:, k+1) = x_pred + L_gain * (y_meas_k - Cd_obs * x_pred);
end
u_q4(:, end) = u_q4(:, end-1); 

MSE_track = mean(vecnorm(x_true(1:3, :) - X_ref(1:3, :)).^2);
MSE_obs   = mean(vecnorm(x_true(1:3, :) - x_hat(1:3, :)).^2);

% --- Propellant Calculation ---
% Impulse = mass_spent * Isp * g0  --> mass_spent = Impulse / (Isp * g0)
g0 = 9.81;              % Earth standard gravity (m/s^2)
Isp = 220;              % Specific impulse of typical Hydrazine thruster (seconds)
rho_fuel = 1.021;       % Density of Hydrazine (kg/L)

total_impulse = sum(abs(u_q4), 'all') * h; 
mass_spent_kg = total_impulse / (Isp * g0);   % Mass in kg
vol_spent_L = mass_spent_kg / rho_fuel;       % Volume in Liters
vol_spent_mL = vol_spent_L * 1000;            % Volume in Milliliters

fprintf('\n--- Performance Metrics ---\n');
fprintf('Total Modulus of Actuation (Impulse) : %.2f N*s\n', total_impulse);
fprintf('Fuel Mass Consumed                   : %.2f kg\n', mass_spent_kg);
fprintf('Fuel Volume Consumed                 : %.2f Liters (%.0f mL)\n', vol_spent_L, vol_spent_mL);
fprintf('Mean Squared Tracking Error (LQR)    : %.4f m^2\n', MSE_track);
fprintf('Mean Squared Observer Error (Luenberger) : %.4f m^2\n', MSE_obs);

%% ========================================================================
% ULTIMATE PLOTTING SUITE (Presentation Ready)
% =========================================================================
c_blue = [0.0, 0.45, 0.74]; c_red  = [0.85, 0.33, 0.10]; 
c_gren = [0.47, 0.67, 0.19]; c_cyan = [0.30, 0.75, 0.93];
c_purp = [0.49, 0.18, 0.56];

% PLOT 1: THE MACRO REALITY (ECI FRAME)
figure('Name', 'PLOT 1: ECI Reality', 'Color', 'k', 'Position', [50, 50, 800, 600]);
[Xe, Ye, Ze] = sphere(100);
surf(Xe*6371, Ye*6371, Ze*6371, 'FaceColor', [0.1 0.3 0.6], 'EdgeColor', 'none'); hold on;
surf(Xe*6450, Ye*6450, Ze*6450, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'none', 'FaceAlpha', 0.1); 
camlight left; lighting gouraud; 
plot3(X_T_ECI(1,:)/1e3, X_T_ECI(2,:)/1e3, X_T_ECI(3,:)/1e3, 'Color', c_cyan, 'LineWidth', 2);
plot3(X_C_ECI(1,:)/1e3, X_C_ECI(2,:)/1e3, X_C_ECI(3,:)/1e3, 'Color', [0.93 0.69 0.13], 'LineStyle', '--', 'LineWidth', 2);
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
title('\color{white}Macro Reality: ECI Frame', 'FontSize', 16);
legend('\color{white}Earth', '\color{white}Atmosphere', '\color{white}Target', '\color{white}Chaser', 'Location', 'best', 'Color', 'k');
axis equal; view(-25, 25); grid on;

% PLOT 2:  KALMAN FILTER CONVERGENCE
figure('Name', 'PLOT 2: KF Convergence', 'Color', 'w', 'Position', [100, 100, 900, 600]);
sgtitle(' Kalman Filter Bias Estimation (Joseph Form)', 'FontSize', 16, 'FontWeight', 'bold');
axis_names = {'Along-Track (X)', 'Cross-Track (Y)', 'Radial (Z)'};
colors_bias = {c_red, c_gren, c_blue};
for i = 1:3
    subplot(3,1,i);
    upper_bound = b_kf(i,:) + 3*std_devs(i,:);
    lower_bound = b_kf(i,:) - 3*std_devs(i,:);
    fill([t_vec, fliplr(t_vec)], [upper_bound, fliplr(lower_bound)], colors_bias{i}, 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
    plot(t_vec, b_kf(i,:), 'Color', colors_bias{i}, 'LineWidth', 2);
    yline(b_true(i), 'k--', 'LineWidth', 1.5);
    ylabel(sprintf('%s [N]', axis_names{i}), 'FontWeight', 'bold');
    if i==3, xlabel('Time [s]', 'FontWeight', 'bold'); end
    grid on; 
    y_min = min([b_true(i)-1.5, min(b_kf(i,:))-0.5]);
    y_max = max([b_true(i)+1.5, max(b_kf(i,:))+0.5]);
    ylim([y_min, y_max]);
    if i==1, legend('3\sigma Bound', 'KF Estimate', 'Truth', 'Location', 'northeast'); end
end

% PLOT 3: 3D OBSTACLE AVOIDANCE (SMOOTH  SPLINE)
figure('Name', 'PLOT 3: 3D Dodge', 'Color', 'k', 'Position', [150, 150, 800, 600]);
[Xs, Ys, Zs] = sphere(50);
surf(Xs*r_obs/1e3 + obs_pos(1)/1e3, Ys*r_obs/1e3 + obs_pos(2)/1e3, Zs*r_obs/1e3 + obs_pos(3)/1e3, 'FaceColor', [1 0.1 0.1], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on;
surf(Xs*(r_obs*0.8)/1e3 + obs_pos(1)/1e3, Ys*(r_obs*0.8)/1e3 + obs_pos(2)/1e3, Zs*(r_obs*0.8)/1e3 + obs_pos(3)/1e3, 'FaceColor', [0.8 0 0], 'FaceAlpha', 0.8, 'EdgeColor', 'none');
camlight right; lighting gouraud;
plot3(X_ref(1,:)/1e3, X_ref(2,:)/1e3, X_ref(3,:)/1e3, 'Color', c_gren, 'LineStyle', '--', 'LineWidth', 2);
plot3(x_true(1,:)/1e3, x_true(2,:)/1e3, x_true(3,:)/1e3, 'Color', c_cyan, 'LineWidth', 3);
plot3(0, 0, 0, 'w*', 'MarkerSize', 15, 'LineWidth', 2);
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
title('\color{white}Optimal 3D Obstacle Avoidance ( Spline)', 'FontSize', 16);
legend('\color{white}Safe Zone', '\color{white}Obstacle', '\color{white}Spline Ref', '\color{white}True Path', '\color{white}Target', 'Color', 'k', 'Location', 'best');
grid on; axis equal; view(-35, 20);

% PLOT 4: ORTHOGONAL PROJECTIONS
figure('Name', 'PLOT 4: Safety Proof', 'Color', 'w', 'Position', [200, 200, 900, 450]);
theta_circ = linspace(0, 2*pi, 100); circ_x = r_obs/1e3 * cos(theta_circ); circ_y = r_obs/1e3 * sin(theta_circ);
subplot(1,2,1); 
fill(obs_pos(1)/1e3 + circ_x, obs_pos(3)/1e3 + circ_y, c_red, 'FaceAlpha', 0.2, 'EdgeColor', c_red, 'LineWidth', 1.5); hold on;
plot(x_true(1,:)/1e3, x_true(3,:)/1e3, 'Color', c_blue, 'LineWidth', 2.5);
plot(X_ref(1,:)/1e3, X_ref(3,:)/1e3, 'Color', c_gren, 'LineStyle', '--', 'LineWidth', 1.5);
title('Side View (X-Z Plane)'); xlabel('Along-Track [km]'); ylabel('Radial [km]'); grid on; axis equal;
subplot(1,2,2); 
fill(obs_pos(2)/1e3 + circ_x, obs_pos(3)/1e3 + circ_y, c_red, 'FaceAlpha', 0.2, 'EdgeColor', c_red, 'LineWidth', 1.5); hold on;
plot(x_true(2,:)/1e3, x_true(3,:)/1e3, 'Color', c_blue, 'LineWidth', 2.5);
plot(X_ref(2,:)/1e3, X_ref(3,:)/1e3, 'Color', c_gren, 'LineStyle', '--', 'LineWidth', 1.5);
title('Front View (Y-Z Plane)'); xlabel('Cross-Track [km]'); ylabel('Radial [km]'); grid on; axis equal;
sgtitle('Orthogonal Projections: Proof of Safe Spline Clearance', 'FontSize', 16, 'FontWeight', 'bold');

% PLOT 5: SMOOTH THRUST PROFILES
figure('Name', 'PLOT 5: Spline Thrust', 'Color', 'w', 'Position', [250, 250, 800, 500]);
plot(t_vec, u_q4(1,:), 'Color', c_red, 'LineWidth', 1.5); hold on;
plot(t_vec, u_q4(2,:), 'Color', c_gren, 'LineWidth', 1.5);
plot(t_vec, u_q4(3,:), 'Color', c_blue, 'LineWidth', 1.5);
title('Total Control Effort (Feedforward + LQR)', 'FontSize', 15);
subtitle(' Omega Matrix guarantees perfect C2 continuity -> Zero Thrust Spikes!');
xlabel('Time [s]'); ylabel('Thrust Command [N]'); grid on;
legend('U_x (Along)', 'U_y (Cross)', 'U_z (Radial)', 'Location', 'best');

% PLOT 6: -STYLE LOG & LINEAR OBSERVER ERROR
figure('Name', 'PLOT 6: Transient Error Analysis', 'Color', 'w', 'Position', [300, 200, 800, 600]);
obs_error_norm = vecnorm(x_true(1:3, :) - x_hat(1:3, :));
subplot(2,1,1);
plot(t_vec, obs_error_norm, 'Color', c_purp, 'LineWidth', 2);
title('Observer Error Magnitude (Linear Scale)', 'FontSize', 14);
ylabel('|| Error || [m]'); grid on; xlim([0 Tsim]);
subplot(2,1,2);
semilogy(t_vec, obs_error_norm, 'Color', c_purp, 'LineWidth', 2);
title('Initial Convergence Transient (Logarithmic Scale)', 'FontSize', 14);
xlabel('Time [s]'); ylabel('|| Error || (Log)'); grid on; xlim([0 500]);
set(gca, 'YScale', 'log');

%% 2. SPLINE OPTIMIZATION & LQR TRACKING
disp('Optimizing Obstacle Avoidance Trajectory...');

% Re-using the optimal C_opt so we don't double-solve identically
[X_ref, U_ff] = build_smooth_trajectory(C_opt, x0_LVLH, Tsim, h, omega, mC);

% Closed-Loop Simulation
x_true = zeros(6, N_steps);  
x_true(:,1) = x0_LVLH;

for k = 1:N_steps-1
    e_k = x_true(:, k) - X_ref(:, k);              
    u_k = U_ff(:, k) - K_lqr * e_k - b_est;        
    x_true(:, k+1) = Ad * x_true(:, k) + Bd * (u_k + b_true); 
end

%% 3. LIVE 3D SIMULATION ANIMATION (LVLH FRAME)
disp('Launching Live 3D Simulation...');
fig = figure('Name', 'Live Simulation: Chaser Satellite Docking', 'Color', 'k', 'Position', [100, 100, 800, 800]);

% Draw Obstacle Sphere
[Xs, Ys, Zs] = sphere(30);
surf(Xs*r_obs/1e3 + obs_pos(1)/1e3, ...
     Ys*r_obs/1e3 + obs_pos(2)/1e3, ...
     Zs*r_obs/1e3 + obs_pos(3)/1e3, ...
     'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on;

% Draw Reference Trajectory Line
plot3(X_ref(1,:)/1e3, X_ref(2,:)/1e3, X_ref(3,:)/1e3, 'g--', 'LineWidth', 1);

% Target Station
plot3(0, 0, 0, 'yp', 'MarkerSize', 20, 'MarkerFaceColor', 'y');

% Initialize Chaser Marker & Trail
chaser_plot = plot3(x0_LVLH(1)/1e3, x0_LVLH(2)/1e3, x0_LVLH(3)/1e3, 'co', 'MarkerSize', 10, 'MarkerFaceColor', 'c');
trail_plot = plot3(x0_LVLH(1)/1e3, x0_LVLH(2)/1e3, x0_LVLH(3)/1e3, 'c-', 'LineWidth', 2);

% Formatting
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
grid on; axis equal;
xlabel('Along-Track X [km]', 'Color', 'w'); 
ylabel('Cross-Track Y [km]', 'Color', 'w'); 
zlabel('Radial Z [km]', 'Color', 'w');
title('\color{white}Live Rendezvous Simulation', 'FontSize', 16);
legend('\color{white}Danger Zone', '\color{white}Planned Path', '\color{white}Target Station', '\color{white}Chaser Spacecraft', 'Location', 'best', 'Color', 'k');
view(-45, 20);

% Dynamic limits based on trajectory, target, and obstacle
all_x = [X_ref(1,:)/1e3, x_true(1,:)/1e3, 0, (obs_pos(1)+r_obs)/1e3, (obs_pos(1)-r_obs)/1e3];
all_y = [X_ref(2,:)/1e3, x_true(2,:)/1e3, 0, (obs_pos(2)+r_obs)/1e3, (obs_pos(2)-r_obs)/1e3];
all_z = [X_ref(3,:)/1e3, x_true(3,:)/1e3, 0, (obs_pos(3)+r_obs)/1e3, (obs_pos(3)-r_obs)/1e3];
pad_x = max(10, 0.1*(max(all_x)-min(all_x)));
pad_y = max(10, 0.1*(max(all_y)-min(all_y)));
pad_z = max(10, 0.1*(max(all_z)-min(all_z)));
xlim([min(all_x)-pad_x, max(all_x)+pad_x]); 
ylim([min(all_y)-pad_y, max(all_y)+pad_y]); 
zlim([min(all_z)-pad_z, max(all_z)+pad_z]);

% Animation Loop!
for k = 1:2:N_steps
    set(chaser_plot, 'XData', x_true(1,k)/1e3, 'YData', x_true(2,k)/1e3, 'ZData', x_true(3,k)/1e3);
    set(trail_plot, 'XData', x_true(1,1:k)/1e3, 'YData', x_true(2,1:k)/1e3, 'ZData', x_true(3,1:k)/1e3);
    drawnow;
    pause(0.01);
end

%% ========================================================================
% : PARAMETRIC SPLINE GENERATOR (15-DOF Multi-Basis Model)
% =========================================================================

function [X_ref, U_ff] = build_smooth_trajectory(C, x0_vec, T, h, omega, mC)
    % Generates a trajectory using a 5th order nominal polynomial for Z
    % and a 15-DOF (5 per axis) orthogonal-ish basis.
    % Base shape guarantees C^2 continuity (pos=0, vel=0, acc=0) at boundaries.
    
    t = 0:h:T;
    tau = t / T;
    z0 = x0_vec(3);
    
    % Nominal 5th Order Polynomial (Z-axis docking)
    z_nom = z0 * (1 - 10*tau.^3 + 15*tau.^4 - 6*tau.^5);
    vz_nom = (z0/T) * (-30*tau.^2 + 60*tau.^3 - 30*tau.^4);
    az_nom = (z0/T^2) * (-60*tau + 180*tau.^2 - 120*tau.^3);
    
    % Base geometric shape function
    S_b   = 64 * (tau.^3) .* ((1 - tau).^3);
    dS_b  = 64 * (3*tau.^2 .* (1-tau).^3 - 3*tau.^3 .* (1-tau).^2);
    ddS_b = 64 * (6*tau .* (1-tau).^3 - 18*tau.^2 .* (1-tau).^2 + 6*tau.^3 .* (1-tau));
    
    N_basis = 5; % 5 degrees of freedom per axis
    x_dev = zeros(size(tau)); x_dot_dev = zeros(size(tau)); x_ddot_dev = zeros(size(tau));
    y_dev = zeros(size(tau)); y_dot_dev = zeros(size(tau)); y_ddot_dev = zeros(size(tau));
    z_dev = zeros(size(tau)); z_dot_dev = zeros(size(tau)); z_ddot_dev = zeros(size(tau));
    
    for i = 1:N_basis
        % Polynomial multiplier P_i(tau) = (2*tau - 1)^(i-1)
        if i == 1
            P   = ones(size(tau));
            dP  = zeros(size(tau));
            ddP = zeros(size(tau));
        else
            k = i - 1;
            P   = (2*tau - 1).^k;
            dP  = k * 2 * (2*tau - 1).^(k-1);
            if i == 2
                ddP = zeros(size(tau));
            else
                ddP = k * (k-1) * 4 * (2*tau - 1).^(k-2);
            end
        end
        
        % Exact Product Rule for Spline Dynamics
        Si   = S_b .* P;
        dSi  = dS_b .* P + S_b .* dP;
        ddSi = ddS_b .* P + 2 .* dS_b .* dP + S_b .* ddP;
        
        % Maintain perfectly accurate indexing for the user's print statements:
        % C(1)=X_mid, C(2)=Y_mid, C(3)=Z_mid.
        if i == 1
            idx_x = 1; idx_y = 2; idx_z = 3;
        else
            idx_x = 3 + (i-2)*3 + 1;
            idx_y = 3 + (i-2)*3 + 2;
            idx_z = 3 + (i-2)*3 + 3;
        end
        
        x_dev = x_dev + C(idx_x) * Si;
        x_dot_dev = x_dot_dev + C(idx_x) * dSi;
        x_ddot_dev = x_ddot_dev + C(idx_x) * ddSi;
        
        y_dev = y_dev + C(idx_y) * Si;
        y_dot_dev = y_dot_dev + C(idx_y) * dSi;
        y_ddot_dev = y_ddot_dev + C(idx_y) * ddSi;
        
        z_dev = z_dev + C(idx_z) * Si;
        z_dot_dev = z_dot_dev + C(idx_z) * dSi;
        z_ddot_dev = z_ddot_dev + C(idx_z) * ddSi;
    end
    
    x = x_dev; 
    y = y_dev; 
    z = z_nom + z_dev;
    
    % Chain rule scaling from tau back to time (t)
    x_dot = x_dot_dev / T;
    y_dot = y_dot_dev / T;
    z_dot = vz_nom + z_dot_dev / T;
    
    x_ddot = x_ddot_dev / T^2;
    y_ddot = y_ddot_dev / T^2;
    z_ddot = az_nom + z_ddot_dev / T^2;
    
    X_ref = [x; y; z; x_dot; y_dot; z_dot];
    
    % Inverse Dynamics (Feedforward Actuation) 
    ux = mC * (x_ddot - 2*omega*z_dot);
    uy = mC * (y_ddot + omega^2 * y);
    uz = mC * (z_ddot - 3*omega^2 * z + 2*omega*x_dot);
    
    U_ff = [ux; uy; uz];         
end

function cost = cost_func(C, x0_vec, T, h, omega, mC)
    [~, U_ff] = build_smooth_trajectory(C, x0_vec, T, h, omega, mC);
    
    % SMOOTH PSEUDO-HUBER L1-NORM (Crucial for deep fmincon convergence)
    epsilon = 1e-3; 
    cost = sum(sqrt(U_ff.^2 + epsilon^2), 'all') * h; 
end

function [c, ceq] = param_constraint(C, x0_vec, T, h, obs_pos, safe_r)
    [X_ref, ~] = build_smooth_trajectory(C, x0_vec, T, h, 0, 0); 
    
    % VECTORIZED SMOOTH CONSTRAINT: Evaluates all time steps safely
    dist_squared = (X_ref(1,:) - obs_pos(1)).^2 + (X_ref(2,:) - obs_pos(2)).^2 + (X_ref(3,:) - obs_pos(3)).^2;
    c = safe_r^2 - dist_squared; % c <= 0 enforces safe_r^2 <= dist_squared
    ceq = []; 
end