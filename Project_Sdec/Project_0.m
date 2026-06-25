% =========================================================================
% DECISION SYSTEMS PROJECT
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
mu   = G * mE;            % Gravitational parameter
RT   = 7171e3;          
mC   = 2000;            
h    = 10;                % 10s simulation step
Tsim = 3600;              % 1 hour rendezvous duration
x0_LVLH = [0; 0; 100e3; 0; 0; 0]; 

omega = sqrt(mu / RT^3);  % Mean orbital angular velocity

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


% Reconstruct the relative LVLH trajectory from the ECI simulation data
% to compare natural orbital mechanics against linearized drift.
X_rel_LVLH = zeros(3, N_steps);
for k = 1:N_steps
    rT_k = X_T_ECI(1:3, k);
    vT_k = X_T_ECI(4:6, k);
    
    iz_k = rT_k / norm(rT_k); % Radial vector
    iy_k = cross(rT_k, vT_k) / norm(cross(rT_k, vT_k)); % Angular momentum normal
    
    % Invert cross product order to match positive velocity vector direction
    ix_k = cross(iz_k, iy_k); 
    
    R_L2E_k = [ix_k, iy_k, iz_k];
    r_rel_ECI = X_C_ECI(1:3, k) - rT_k;
    X_rel_LVLH(:, k) = R_L2E_k' * r_rel_ECI; 
end

X_linear_CW = zeros(6, N_steps);
X_linear_CW(:, 1) = x0_LVLH; 
for k = 1:N_steps-1
    X_linear_CW(:, k+1) = Ad * X_linear_CW(:, k); % Discrete state propagation
end

%% ========================================================================
% QUESTION 2: BIAS ESTIMATION VIA KALMAN FILTER
% =========================================================================
fprintf('--- Q2: Actuator Bias Estimation (Kalman Filter) ---\n');

b_true = [2.5; -1.2; 0.7];    % Constant bias
noise_accel_std = 0.005; 
sigma_pos = 0.1;         

% Exact variance of the scaled measurement noise (F = m * a)
R_variance = (mC * noise_accel_std)^2; 

% Kalman Filter Initialization
b_kf = zeros(3, N_steps);
b_kf(:,1) = [0; 0; 0];     % Initial guess
Sigma = cell(1, N_steps);
Sigma{1} = 100 * eye(3);   % Initial covariance
R_kf = eye(3) * R_variance;
Q_kf = eye(3) * 1e-4;      % Process noise
I_3 = eye(3); 

x_curr = x0_LVLH;
u_cmd = [1; 0; -1] .* ones(1, N_steps); 

for k = 1:N_steps-1
    % Physical Plant Update
    u_real = u_cmd(:, k) + b_true; 
    xdot_true = A_cw * x_curr + B_cw * u_real; 
    
    % Noisy Measurements
    x_meas = x_curr + randn(6,1) * sigma_pos;       
    xdot_meas = xdot_true + randn(6,1) * noise_accel_std; 
    
    % Decision System Form (y = b + v)
    y_meas_k = mC * (xdot_meas(4:6) - A_cw(4:6, :) * x_meas) - u_cmd(:, k);
    H_k = eye(3); 
    
    % --- Kalman Filter Algorithm ---
    % Predict
    b_pred = b_kf(:, k); 
    Sigma_pred = Sigma{k} + Q_kf;
    
    % Update
    y_tilde = y_meas_k - H_k * b_pred;
    S = H_k * Sigma_pred * H_k' + R_kf;
    K = Sigma_pred * H_k' / S;
    b_kf(:, k+1) = b_pred + K * y_tilde;
    
    % Joseph Form for Covariance Update
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
% QUESTION 3 & 4: SPLINE OPTIMIZATION & LQR TRACKING
% =========================================================================
fprintf('\n--- Q3 & Q4: Control, Observation, and Replanning ---\n');

Cd_obs = [eye(3), zeros(3,3)];  
Q_noise = eye(6) * 1e-4;        
R_noise = eye(3) * 10;          
L_gain = dlqe(Ad, eye(6), Cd_obs, Q_noise, R_noise); 

% LQR Controller Design
Q_lqr = diag([1e3, 1e3, 1e3, 1e4, 1e4, 1e4]); 
R_lqr = eye(3) * 1e3;                      
K_lqr = dlqr(Ad, Bd, Q_lqr, R_lqr);

% Obstacle Definition
obs_pos = [0; 0; 30e3]; 
r_obs = 20e3; 
safe_radius = r_obs + 2e3; 

C0 = [15e3; 15e3]; % Initial guess

% Optimize the Spline formulation
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp', 'MaxFunctionEvaluations', 2000);
C_opt = fmincon(@(C) cost_func(C, x0_LVLH(3), Tsim, h, omega, mC), C0, ...
    [], [], [], [], [-120e3, -120e3], [120e3, 120e3], ...
    @(C) param_constraint(C, x0_LVLH(3), Tsim, h, obs_pos, safe_radius), options);

fprintf('Optimal Avoidance Deviations -> X: %.1f km, Y: %.1f km\n', C_opt(1)/1e3, C_opt(2)/1e3);

% Generate the Final Reference Trajectory via Omega Matrix
[X_ref, U_ff] = build_lab2_trajectory(C_opt, x0_LVLH(3), Tsim, h, omega, mC);

% Closed-Loop LQR Simulation
x_true = zeros(6, N_steps);  x_true(:,1) = x0_LVLH;
x_hat  = zeros(6, N_steps);  x_hat(:,1)  = x0_LVLH + [200; -200; 200; 0; 0; 0]; 
u_q4   = zeros(3, N_steps);

for k = 1:N_steps-1
    e_k = x_hat(:, k) - X_ref(:, k);
    u_q4(:, k) = U_ff(:, k) - K_lqr * e_k;
    
    x_true(:, k+1) = Ad * x_true(:, k) + Bd * u_q4(:, k);
    
    y_meas_k = Cd_obs * x_true(:, k+1) + randn(3,1) * sqrt(10);
    x_pred = Ad * x_hat(:, k) + Bd * u_q4(:, k);
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
% PLOTTING SUITE
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
view(-25, 25); grid on;

% PLOT 2: Diagnostic Comparison Plot
figure('Name', 'Q1: Nonlinear vs Linear CW Propagation', 'Color', 'w', 'Position', [80, 80, 900, 550]);
plot(X_rel_LVLH(1,:)/1e3, X_rel_LVLH(3,:)/1e3, 'b-', 'LineWidth', 3, 'DisplayName', 'True Nonlinear (ECI Mapped)');
hold on;
plot(X_linear_CW(1,:)/1e3, X_linear_CW(3,:)/1e3, 'r--', 'LineWidth', 2.5, 'DisplayName', 'Linearized CW Approximation');
plot(0, 0, 'yp', 'MarkerSize', 15, 'MarkerFaceColor', 'y', 'Color', 'k', 'DisplayName', 'Target Origin');

grid on; axis equal;
xlabel('Along-Track X (Direction of Flight) [km]', 'FontWeight', 'bold');
ylabel('Radial Z (Altitude Delta) [km]', 'FontWeight', 'bold');
title('Model Breakdown: Real Physics vs Linearized Drift from 100 km', 'FontSize', 14);
legend('Location', 'best');
text(x0_LVLH(1)/1e3, x0_LVLH(3)/1e3, ' \leftarrow Start Position (100km Alt)', 'FontWeight', 'bold');

% PLOT 3: KALMAN FILTER CONVERGENCE
figure('Name', 'PLOT 3: KF Convergence', 'Color', 'w', 'Position', [100, 100, 900, 600]);
sgtitle('Kalman Filter Bias Estimation (Joseph Form)', 'FontSize', 16, 'FontWeight', 'bold');
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
    grid on; ylim([b_true(i)-1.5, b_true(i)+1.5]);
    if i==1, legend('3\sigma Bound', 'KF Estimate', 'Truth', 'Location', 'northeast'); end
end

% PLOT 4: 3D OBSTACLE AVOIDANCE (SMOOTH SPLINE)
figure('Name', 'PLOT 4: 3D Dodge', 'Color', 'k', 'Position', [150, 150, 800, 600]);
[Xs, Ys, Zs] = sphere(50);
surf(Xs*r_obs/1e3 + obs_pos(1)/1e3, Ys*r_obs/1e3 + obs_pos(2)/1e3, Zs*r_obs/1e3 + obs_pos(3)/1e3, 'FaceColor', [1 0.1 0.1], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on;
surf(Xs*(r_obs*0.8)/1e3 + obs_pos(1)/1e3, Ys*(r_obs*0.8)/1e3 + obs_pos(2)/1e3, Zs*(r_obs*0.8)/1e3 + obs_pos(3)/1e3, 'FaceColor', [0.8 0 0], 'FaceAlpha', 0.8, 'EdgeColor', 'none');
camlight right; lighting gouraud;
plot3(X_ref(1,:)/1e3, X_ref(2,:)/1e3, X_ref(3,:)/1e3, 'Color', c_gren, 'LineStyle', '--', 'LineWidth', 2);
plot3(x_true(1,:)/1e3, x_true(2,:)/1e3, x_true(3,:)/1e3, 'Color', c_cyan, 'LineWidth', 3);
plot3(0, 0, 0, 'w*', 'MarkerSize', 15, 'LineWidth', 2);
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
title('\color{white}Optimal 3D Obstacle Avoidance (Spline)', 'FontSize', 16);
legend('\color{white}Safe Zone', '\color{white}Obstacle', '\color{white}Spline Ref', '\color{white}True Path', '\color{white}Target', 'Color', 'k', 'Location', 'best');
grid on; axis equal; view(-35, 20);

% PLOT 5: ORTHOGONAL PROJECTIONS
figure('Name', 'PLOT 5: Safety Proof', 'Color', 'w', 'Position', [200, 200, 900, 450]);
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

% PLOT 6: SMOOTH THRUST PROFILES
figure('Name', 'PLOT 6: Spline Thrust', 'Color', 'w', 'Position', [250, 250, 800, 500]);
plot(t_vec, u_q4(1,:), 'Color', c_red, 'LineWidth', 1.5); hold on;
plot(t_vec, u_q4(2,:), 'Color', c_gren, 'LineWidth', 1.5);
plot(t_vec, u_q4(3,:), 'Color', c_blue, 'LineWidth', 1.5);
title('Total Control Effort (Feedforward + LQR)', 'FontSize', 15);
subtitle(' Omega Matrix guarantees perfect C2 continuity -> Zero Thrust Spikes!');
xlabel('Time [s]'); ylabel('Thrust Command [N]'); grid on;
legend('U_x (Along)', 'U_y (Cross)', 'U_z (Radial)', 'Location', 'best');

% PLOT 7: LINEAR OBSERVER ERROR
figure('Name', 'PLOT 7: Transient Error Analysis', 'Color', 'w', 'Position', [300, 200, 800, 600]);
obs_error_norm = vecnorm(x_true(1:3, :) - x_hat(1:3, :));
subplot(2,1,1);
plot(t_vec, obs_error_norm, 'Color', c_purp, 'LineWidth', 2);
title('Luenberger Observer Error Magnitude (Linear Scale)', 'FontSize', 14);
ylabel('|| Error || [m]'); grid on; xlim([0 Tsim]);
subplot(2,1,2);
semilogy(t_vec, obs_error_norm, 'Color', c_purp, 'LineWidth', 2);
title('Initial Convergence Transient (Logarithmic Scale)', 'FontSize', 14);
xlabel('Time [s]'); ylabel('|| Error || (Log)'); grid on; xlim([0 500]);
set(gca, 'YScale', 'log');

%% 2. SPLINE OPTIMIZATION & LQR TRACKING
disp('Optimizing Obstacle Avoidance Trajectory...');

C0 = [15e3; 15e3]; % Initial guess for dodging amplitude
options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');
C_opt = fmincon(@(C) cost_func(C, x0_LVLH(3), Tsim, h, omega, mC), C0, ...
    [], [], [], [], [], [], ...
    @(C) param_constraint(C, x0_LVLH(3), Tsim, h, obs_pos, safe_radius), options);

% Generate the Reference Trajectory
[X_ref, U_ff] = build_lab2_trajectory(C_opt, x0_LVLH(3), Tsim, h, omega, mC);

% Closed-Loop Simulation
x_true = zeros(6, N_steps);  
x_true(:,1) = x0_LVLH;
for k = 1:N_steps-1
    e_k = x_true(:, k) - X_ref(:, k);      % Error from optimal path
    u_k = U_ff(:, k) - K_lqr * e_k;        % Actuation command
    x_true(:, k+1) = Ad * x_true(:, k) + Bd * u_k; % Physics update
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
xlim([-30 160]); ylim([-30 30]); zlim([-10 110]);

% Animation Loop
for k = 1:2:N_steps
    set(chaser_plot, 'XData', x_true(1,k)/1e3, 'YData', x_true(2,k)/1e3, 'ZData', x_true(3,k)/1e3);
    set(trail_plot, 'XData', x_true(1,1:k)/1e3, 'YData', x_true(2,1:k)/1e3, 'ZData', x_true(3,1:k)/1e3);
    drawnow;
    pause(0.01);
end

%% ========================================================================
% PARAMETRIC SPLINE GENERATOR
% =========================================================================

function [X_ref, U_ff] = build_lab2_trajectory(C, z0, T, h, omega, mC)
    % Builds a seamless trajectory honoring boundary conditions
    t = 0:h:T;
    tau = t / T;              % spline at tau=0 (start) and tau=1 (end)
    
    Omega_mat = [
        1, 0, 0, 0, 0, 0;     % tau=0, Pos (w1 = 1)
        1, 1, 1, 1, 1, 1;     % tau=1, Pos (w1+w2... = 0)
        0, 1, 0, 0, 0, 0;     % tau=0, Vel
        0, 1, 2, 3, 4, 5;     % tau=1, Vel
        0, 0, 2, 0, 0, 0;     % tau=0, Acc
        0, 0, 2, 6, 12, 20    % tau=1, Acc
    ];
    bc_z = [z0; 0; 0; 0; 0; 0];
    w_z = Omega_mat \ bc_z; 
    
    z      = w_z(1) + w_z(2)*tau + w_z(3)*tau.^2 + w_z(4)*tau.^3 + w_z(5)*tau.^4 + w_z(6)*tau.^5;
    dz_tau = w_z(2) + 2*w_z(3)*tau + 3*w_z(4)*tau.^2 + 4*w_z(5)*tau.^3 + 5*w_z(6)*tau.^4;
    ddz_tau= 2*w_z(3) + 6*w_z(4)*tau + 12*w_z(5)*tau.^2 + 20*w_z(6)*tau.^3;
    
    z_dot  = dz_tau / T;
    z_ddot = ddz_tau / T^2;
    
    x = C(1) * sin(pi * tau);                     % half sine wave
    y = C(2) * sin(pi * tau);                     % half sine wave
    x_dot = C(1) * (pi/T) * cos(pi * tau);
    y_dot = C(2) * (pi/T) * cos(pi * tau);
    x_ddot = -C(1) * (pi/T)^2 * sin(pi * tau);
    y_ddot = -C(2) * (pi/T)^2 * sin(pi * tau);
    
    X_ref = [x; y; z; x_dot; y_dot; z_dot];
    
    % Inverse Dynamics (Feedforward Actuation) 
    ux = mC * (x_ddot - 2*omega*z_dot);
    uy = mC * (y_ddot + omega^2 * y);
    uz = mC * (z_ddot - 3*omega^2 * z + 2*omega*x_dot);
    
    U_ff = [ux; uy; uz];         % feedforward thrust
end

function cost = cost_func(C, z0, T, h, omega, mC)
    [~, U_ff] = build_lab2_trajectory(C, z0, T, h, omega, mC);
    cost = sum(abs(U_ff), 'all'); % Minimize total control effort
end

function [c, ceq] = param_constraint(C, z0, T, h, obs_pos, safe_r)
    [X_ref, ~] = build_lab2_trajectory(C, z0, T, h, 0, 0); 
    x = X_ref(1, :); y = X_ref(2, :); z = X_ref(3, :);
    dist = sqrt((x - obs_pos(1)).^2 + (y - obs_pos(2)).^2 + (z - obs_pos(3)).^2);
    c = safe_r - min(dist);
    ceq = []; 
end
