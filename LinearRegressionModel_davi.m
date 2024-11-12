restart_system();

data = read_csv('data_4050.csv');
[predictors, targets] = load_array(data);

head(targets);

for column = ["q1", "q2", "q3", "r0"]
    model = fitlm(predictors,targets.(column));
    
    disp(">>>> MODEL: " + column);
    disp(model);

    disp("> COEFFICIENTS: ");
    disp(model.Coefficients.Estimate);               % Coeficientes [intercepto, b1, b2]
    pred_values = predict(model, predictors);           % Predições
    
    disp("> PREDICTED VALUES: ");
    head(pred_values);
    predictors.(column) = pred_values;                  % Alimentar independentes com a predição ou com a fonte

    disp("> ANOVA: ");
    anova(model,'summary')
end

head(predictors);

%plot(mdl)

% DEFINIÇÃO DE FUNÇÕES
% ----------------------------------------------------------- %
function restart_system()
    clear
    close all
    clc
end

% Função para importar arquivos csv
function data = read_csv(file)
    data = readtable(file);
    head(data);
end

% Função para carregar vetores com os dados de entrada
function [predictors, targets] = load_array(data)
    predictors = data(:, {'R_', 'L_', 'C_', 'D_', 'Vi', 'J'});
    targets = data(:, {'q1', 'q2', 'q3', 'r0'});
end

% nominal system
function sys = nominal_system(data)
    A = [-1/(data.R_*data.C_) , (1-data.D_)/data.C_ ; -(1-data.D_)/data.L_ , 0];
    B = [-data.Vi/((1-data.D_)^2*data.R_*data.C_) ; data.Vi/((1-D_)*data.L_)];
    C = [data.iL*(1-data.D_) , data.Vo*(1-data.D_)];
    D = -data.Vo*data.iL;
    sys = ss(A,B,C,D);
end

% Ackermann's gains
function [knom, kinom] = ackermanns_gains()
    Knom  = [-0.0013 , 0.0286];
    Kinom = 0.3982;
end


% controller gain calculation
function [Ks, Ss, Ps, K, Ki] = controller_gain_calculation(sys, Q, R0)
    [Ks,Ss,Ps] = lqi(sys,Q,R0);
    K  = Ks(1:2);
    Ki = -Ks(3);
end

% closed-loop system
function [u, sys_mf] = closedLoop_system()
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
function [Z, a, b, c, d] = step_info(sys_mf)
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