restart_system();

data = read_csv('data_4050.csv');
[predictors, targets] = load_array(data);

for column = ["q1", "q2", "q3", "r0"]
    model = fitlm(predictors,targets.(column));
    
    disp(">>>> MODEL: " + column);
    disp(model);

    disp("> COEFFICIENTS: ");
    disp(model.Coefficients.Estimate);               % Coeficientes [intercepto, b1, b2]
    pred_values = predict(model, predictors);        % Predições
    
    disp("> PREDICTED VALUES: ");
    head(pred_values);
    predictors.(column) = pred_values;                % Alimentar independentes com a predição ou com a fonte

    disp("> ANOVA: ");
    anova(model,'summary');
end

% Ackermann's gains
Knom  = [-0.0013 , 0.0286];
Kinom = 0.3982;

% simulation vectors
dt = 1e-6;
t  = 0:dt:0.1;
r  = ones(length(t),1);

for i = 1:size(predictors.R_)
    [sys, A, B, C, D] = nominal_system(predictors, i);

    % os valores de Q e R devem ser positivos
    Q(1,1) = abs(predictors.q1(i));
    Q(2,2) = abs(predictors.q2(i));
    Q(3,3) = abs(predictors.q3(i));
    R0 = abs(predictors.r0(i));
    % disp("-----------------------<<<<<");
    % disp(Q);

    % 
    [Ks, K, Ki] = controller_gain_calculation(sys, Q, R0);
    [u, sys_mf] = closedLoop_system(A, B, C, D, K, Ki, r, t);
    [penalty] = penalty_control(u, predictors.D_(i));
    [a, b, c, d] = step_info(sys_mf);
    [J] = cost_calculation(a, b, c, d, penalty);
    disp(J);
end


% FUNCTIONS
% ----------------------------------------------------------- %
function restart_system()
    clear
    close all
    clc
end

% import csv file
function data = read_csv(file)
    disp("> reading file...");
    data = readtable(file);
    head(data);
end

% load input data
function [predictors, targets] = load_array(data)
    disp("> loading data...");
    predictors = data(:, {'R_', 'L_', 'C_', 'D_', 'Vi', 'J'});
    targets = data(:, {'q1', 'q2', 'q3', 'r0'});
end

% nominal system
function [sys, A, B, C, D] = nominal_system(data, i)
    iL = 6;
    Vo = 40;

    A = [-1/(data.R_(i)*data.C_(i)) , (1-data.D_(i))/data.C_(i) ; -(1-data.D_(i))/data.L_(i) , 0];
    B = [-data.Vi(i)/((1-data.D_(i))^2*data.R_(i)*data.C_(i)) ; data.Vi(i)/((1-data.D_(i))*data.L_(i))];
    C = [iL*(1-data.D_(i)) , Vo*(1-data.D_(i))];
    D = -Vo*iL;
    sys = ss(A,B,C,D);
end

% controller gain calculation
function [Ks, K, Ki] = controller_gain_calculation(sys, Q, R0) 
    [Ks,Ss,Ps] = lqi(sys,Q,R0);      % ?? ONDE FOI USADO O Ss E O Ps ??
    K  = Ks(1:2);
    Ki = -Ks(3);
end

% closed-loop system
function [u, sys_mf] = closedLoop_system(A, B, C, D, K, Ki, r, t)
    Aa = [A-B*K , B*Ki ; -(C-D*K) , -D*Ki];
    Ba = [0 ; 0 ; 1];
    Ca = [C-D*K , D*Ki];
    sys_mf = ss(Aa,Ba,Ca,0);
    [Y,T,X] = lsim(sys_mf,r,t);      % time simulation
    u = [-K , Ki]*X';                % control signal
end

% Penalize if control signal exceeds the maximum limit
function [penalty] = penalty_control(u, D_)
    if any(abs(u) > D_)
        penalty = 1e5;  % High penalty if the control exceeds limit
    else
        penalty = 0;
    end
end

% step info
function [a, b, c, d] = step_info(sys_mf)
    Z = stepinfo(sys_mf);
    a = Z.RiseTime;
    b = Z.SettlingTime;
    c = Z.Overshoot;
    d = Z.Undershoot;
end

% Calculate the cost function J with weighting factors
function [J] = cost_calculation(a, b, c, d, penalty)

    % weighting factors for the optimization
    % (the larger the more important)
    % sum should be equal 1
    pond1 = 0.2;       % rise time       
    pond2 = 0.4;       % settling time
    pond3 = 0.3;       % overshoot
    pond4 = 0.1;       % undershoot

    J = pond1*a + pond2*b + pond3*c + pond4*d + penalty;
end
