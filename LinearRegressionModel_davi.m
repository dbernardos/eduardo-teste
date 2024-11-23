% Reinicia o sistema (limpa o workspace, fecha figuras e limpa o terminal)
restart_system();  
% Chama a função principal com parâmetros (flag=para gráficos, rep=para qtde)
principal(true, 2);

% ----------------------------------------------------------- %
% FUNÇÃO PRINCIPAL
% ----------------------------------------------------------- %
function principal(flag, rep)
    % Lê os dados do arquivo CSV
    data = read_csv('data_4050.csv');
    % Carrega preditores e alvos (targets)
    [predictors, targets] = load_array(data);
    
    % Inicializa estrutura para armazenar novos valores preditos
    new = [];
    
    % Para cada coluna de saída alvo (q1, q2, q3, r0), ajusta modelos
    for column = ["q1", "q2", "q3", "r0"]
        %lrm_crossValidation(predictors, targets, column, new, data); %new, column, 
        new = lrm_simple(predictors, targets, column, new);
        head(new.(column));
    end

    % application_domain(predictors, data, new, flag, rep);
    
end

% ----------------------------------------------------------- %
% FUNÇÕES AUXILIARES
% ----------------------------------------------------------- %

% Reinicia o ambiente do MATLAB
function restart_system()
    clear;      % Limpa variáveis do workspace
    close all;  % Fecha todas as figuras abertas
    clc;        % Limpa o terminal
end

% Lê um arquivo CSV e retorna uma tabela de dados
function data = read_csv(file)
    disp("> Lendo arquivo...");
    data = readtable(file);     % Lê o arquivo CSV
    head(data);                 % Exibe os primeiros valores para verificação
end

% Carrega preditores (entradas) e alvos (saídas)
function [predictors, targets] = load_array(data)
    disp("> Carregando dados...");
    % Define os preditores e alvos (com base em colunas relevantes)
    predictors = data(:, {'J'});                  % Apenas coluna J como preditor
    targets = data(:, {'q1', 'q2', 'q3', 'r0'});  % Alvos (variáveis dependentes)
end

% Define o sistema nominal para uma dada linha de dados
function [sys, A, B, C, D] = nominal_system(data, i)
    iL = 6;  % Corrente nominal
    Vo = 40; % Tensão nominal

    % Matrizes do sistema (A, B, C, D) calculadas com base nos dados
    A = [-1/(data.R_(i)*data.C_(i)) , (1-data.D_(i))/data.C_(i) ; -(1-data.D_(i))/data.L_(i) , 0];
    B = [-data.Vi(i)/((1-data.D_(i))^2*data.R_(i)*data.C_(i)) ; data.Vi(i)/((1-data.D_(i))*data.L_(i))];
    C = [iL*(1-data.D_(i)) , Vo*(1-data.D_(i))];
    D = -Vo*iL;
    sys = ss(A,B,C,D);
end

% Calcula os ganhos do controlador (K, Ki) com base nas penalidades (Q, R0)
function [Ks, K, Ki] = controller_gain_calculation(sys, Q, R0)
    [Ks,Ss,Ps] = lqi(sys,Q,R0); % Calcula os ganhos do controlador (LQI)
    K  = Ks(1:2);
    Ki = -Ks(3);
end

% Simula o sistema em malha fechada
function [u, sys_mf] = closedLoop_system(A, B, C, D, K, Ki, r, t)
    % Define as matrizes do sistema em malha fechada
    Aa = [A-B*K , B*Ki ; -(C-D*K) , -D*Ki];
    Ba = [0 ; 0 ; 1];
    Ca = [C-D*K , D*Ki];
    sys_mf = ss(Aa, Ba, Ca, 0);

    % Simula a resposta do sistema e calcula o sinal de controle
    [Y,T,X] = lsim(sys_mf,r,t);  % Simulação no tempo
    u = [-K , Ki]*X';            % Calcula o sinal de controle
end

% Penaliza o controle se ultrapassar limites
function [penalty] = penalty_control(u, D_)
    if any(abs(u) > D_)
        penalty = 1e5;  % Penalidade alta para violação
    else
        penalty = 0;
    end
end

% Extrai informações sobre a resposta
function [a, b, c, d] = step_info(sys_mf)
    Z = stepinfo(sys_mf);  % Calcula informações de resposta
    a = Z.RiseTime;        % Tempo de subida
    b = Z.SettlingTime;    % Tempo de acomodação
    c = Z.Overshoot;       % Sobressinal
    d = Z.Undershoot;      % Suboscilações
end

% Calcula a função de custo com base nas métricas e penalidades
function [J] = cost_calculation(a, b, c, d, penalty)
    % Pesos de importância para cada métrica
    pond1 = 0.2;  % Tempo de subida
    pond2 = 0.4;  % Tempo de acomodação
    pond3 = 0.3;  % Sobressinal
    pond4 = 0.1;  % Suboscilações

    % Função de custo ponderada
    J = pond1*a + pond2*b + pond3*c + pond4*d + penalty;
end

% Ajusta um modelo linear simples para cada coluna-alvo (q1, q2, q3, r0)
function new = lrm_simple(predictors, targets, column, new)
    % Ajusta um modelo linear aos dados
    model = fitlm(predictors, targets.(column));
    
    % Exibe informações sobre o modelo ajustado
    disp(">>>> MODELO: " + column);
    disp(model);
    
    % Exibe os coeficientes do modelo (intercepto e inclinações)
    disp("> COEFICIENTES: ");
    disp(model.Coefficients.Estimate);  % Vetor de coeficientes
    
    % Gera predições para os preditores fornecidos
    pred_values = predict(model, predictors);

    % Exibe as primeiras predições (aplicando limite mínimo de 0.001)
    disp("> VALORES PREVISTOS: ");
    head(max(0.001, pred_values));

    % Armazena as predições no vetor `new` correspondente à coluna-alvo
    new.(column) = max(0.001, pred_values);

    % Exibe a análise de variância (ANOVA) do modelo
    disp("> ANOVA: ");
    anova(model, 'summary');

    %plot(model);
end

% Regra
function application_domain(predictors, data, new, flag, rep)
    % Ganhos nominais de Ackermann (valores conhecidos para comparação)
    Knom  = [-0.0013 , 0.0286];
    Kinom = 0.3982;

    % Configurações para simulação (vetor de tempo e referência)
    dt = 1e-6;                % Passo de tempo
    t  = 0:dt:0.1;            % Intervalo de tempo
    r  = ones(length(t), 1);  % Referência constante (1)

    % Loop sobre cada instância (linha) dos preditores
    for i = 1:size(predictors.J)
        % Define o sistema nominal para a i-ésima linha dos dados
        [sys, A, B, C, D] = nominal_system(data, i);

        % Preenche matrizes de penalidade (Q) e R0 com valores preditos
        Q(1,1) = new.q1(i);
        Q(2,2) = new.q2(i);
        Q(3,3) = new.q3(i);
        R0 = new.r0(i);

        % Calcula os ganhos do controlador com base no sistema e penalidades
        [Ks, K, Ki] = controller_gain_calculation(sys, Q, R0);

        % Simula o sistema em malha fechada
        [u, sys_mf] = closedLoop_system(A, B, C, D, K, Ki, r, t);

        % Aplica penalidade se o sinal de controle ultrapassar o limite
        [penalty] = penalty_control(u, data.D_(i));

        % Extrai informações de desempenho da resposta ao degrau
        [a, b, c, d] = step_info(sys_mf);

        % Calcula a função de custo J (baseada no desempenho e penalidades)
        [J] = cost_calculation(a, b, c, d, penalty);

        % Se a flag estiver ativada e dentro do número de repetições, plota gráficos
        if flag == true && i <= rep
            plot_chart(new.q1(i), new.q2(i), new.q3(i), new.r0(i), sys, A, B, C, D, Knom, Kinom, r, t);
        end
    end
end


% ----------------------------------------------------------- %
% FUNÇÃO PARA VALIDAÇÃO CRUZADA (CROSS-VALIDATION)
% ----------------------------------------------------------- %
function allPredictions = lrm_crossValidation(predictors, targets, column, new, data)
    % Configuração para validação cruzada
    K = 5;                                       % Número de folds
    cv = cvpartition(height(data), 'KFold', K);  % Divide os dados em K folds
    target = targets.(column);                   % Seleciona a coluna-alvo atual

    % Inicializa vetores para armazenar métricas
    rmse_values = zeros(K, 1);              % Erro quadrático médio (RMSE) em cada fold
    r2_values = zeros(K, 1);                % Coeficiente de determinação (R²) em cada fold
    allPredictions = nan(height(data), 1);  % Armazena todas as predições na ordem original

    % Loop para realizar a validação cruzada em K folds
    for i = 1:K
        % Separa os índices para treino e teste
        trainIdx = training(cv, i);  % Índices do conjunto de treinamento
        testIdx = test(cv, i);       % Índices do conjunto de teste

        % Separa os dados de treinamento e teste
        trainPredictors = predictors(trainIdx, :);
        trainTargets = target(trainIdx);
        testPredictors = predictors(testIdx, :);
        testTargets = target(testIdx);

        % Ajusta o modelo linear com os dados de treinamento
        model = fitlm(trainPredictors, trainTargets);

        % Faz predições no conjunto de teste
        predictedTargets = predict(model, testPredictors);

        % Armazena as predições na ordem original dos dados
        allPredictions(testIdx) = predictedTargets;

        % Calcula o RMSE (erro médio quadrático)
        rmse_values(i) = sqrt(mean((predictedTargets - testTargets).^2));

        % Calcula o coeficiente de determinação (R²)
        ss_residual = sum((testTargets - predictedTargets).^2);  % Soma dos quadrados residuais
        ss_total = sum((testTargets - mean(testTargets)).^2);    % Soma total dos quadrados
        r2_values(i) = 1 - (ss_residual / ss_total);             % R²
    end

    % Exibe as métricas médias (RMSE e R²) após os K folds
    disp("RMSE médio em " + K + " folds: " + mean(rmse_values));
    disp("R² médio em " + K + " folds: " + mean(r2_values));

    % Exibe o vetor de predições reorganizado
    disp("Vetor de predições (ordem original):");
    disp(allPredictions);
end

% ----------------------------------------------------------- %
% FUNÇÃO PARA GERAR GRÁFICOS DE COMPARAÇÃO
% ----------------------------------------------------------- %
function plot_chart(q1, q2, q3, r0, sys, A, B, C, D, Knom, Kinom, r, t)
    % Configura as matrizes de penalidade
    Qbest = diag([q1, q2, q3]);

    % Calcula os ganhos do controlador
    [Ks1,Ss1,Ps1] = lqi(sys,Qbest,r0);
    K1  = Ks1(1:2);
    Ki1 = -Ks1(3);

    % Define o sistema em malha fechada usando os valores otimizados
    Aa1 = [A-B*K1 , B*Ki1 ; -(C-D*K1) , -D*Ki1];
    Ba1 = [0 ; 0 ; 1];
    Ca1 = [C-D*K1 , D*Ki1];
    sysbest = ss(Aa1, Ba1, Ca1, 0); 

    % Define o sistema em malha fechada com os valores nominais de Ackermann
    Aa2 = [A-B*Knom , B*Kinom ; -(C-D*Knom) , -D*Kinom];
    Ba2 = [0 ; 0 ; 1];
    Ca2 = [C-D*Knom , D*Kinom];
    sysnom = ss(Aa2, Ba2, Ca2, 0); 

    % Simula os dois sistemas para a mesma entrada (referência r)
    [Y1, ~, X1] = lsim(sysbest, 3*r, t);  % Resposta do sistema
    u1 = [-K1 , Ki1]*X1';                 % Sinal de controle

    [Y2, ~, X2] = lsim(sysnom, 3*r, t);   % Resposta do sistema nominal
    u2 = [-Knom , Kinom]*X2';             % Sinal de controle (nominal)

    % Plota os resultados da comparação entre os dois sistemas
    figure;
    % Subgráfico 1: Comparação das respostas (potência)
    subplot(211);
    plot(t, Y1, t, Y2);
    grid;
    legend('LR', 'Nominal');
    ylabel('Potência');
    
    % Subgráfico 2: Comparação dos sinais de controle
    subplot(212);
    plot(t, u1, t, u2);
    grid;
    ylabel('Sinal de controle');
    xlabel('Tempo (s)');
end
