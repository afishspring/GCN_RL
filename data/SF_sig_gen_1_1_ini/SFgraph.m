% % % Generate Adjacency Matrix A
% % n = 1000;
% % % p = 0.0075;
% % % A = rand(n,n) < p;
% % % A = triu(A,1);
% % % A = A + A';
% % seed =[0 1 0 0 0;1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0]
% % A = SFNG(n, 4, seed);
% % % ------------------------------------------ Change before this line.
% % W = randi(100,n); % distance, or weight
% % Aw = A .* W;
% % Aw = triu(Aw,1);
% % Aw = Aw + Aw';
% % [NoComp, Size, ~]=networkComponents(A);
% % if NoComp == 1
% %     save('AdjMw.mat',Aw)
% % else
% %     sprintf('The network is not connected, try again.\n')
% % end

%% SI Epidemics on Aw
% Configuration Parameters
load('AdjMw.mat');
NetSize = size(Aw,2);
betaMax = 0.1; % infection rate

tMax = 2000; %max # of time steps
I0 = 5; % No. of seeds

% fileID = fopen('ParaSetting.txt','w');
% fprintf(fileID, 'Network = ER\r\n Distance = [1:100]\r\n Size = %i\r\n p= %f\r\n Beta = %f\r\n I0 =%i\r\n Infection Rate = Beta/Aw', NetSize, p, betaMax, I0);
% fclose(fileID);

% Settings
Beta = betaMax ./ Aw; % Assume infection rate to be inverse of distance
Beta(Beta==inf) = 0;
stub = 100;
MaxRuntime = 10;

    
    % Initialization
    Count = zeros(1,tMax);
    Dist = Aw^5;
    InitState = zeros(1,NetSize);
    [sortV, sortI] = sort(Dist(:), 'descend');
    for i = 1:I0
        Ind = sortI(i);
        [x,y] = ind2sub(size(Dist),Ind);
        InitState(x) = 1;
        InitState(y) = 1;
    end

%   Perm = randperm(NetSize); % for randomly chosen ones
%     for i=1:I0
%         InitState(Perm(i))=1;
%     end
    
% Evolution
State = InitState;
for t = 1:tMax
    TmpState = goOneStep(State, Beta);
    State = TmpState;
    infCnt = sum((State>0));
    if infCnt >= NetSize - 90
        fprintf('Done. Check state vectors.\n')
        break
    end
    fprintf('t = %d, infCnt = %d\n', t, infCnt);
    if infCnt >= stub && infCnt < stub + 100
        % log state vector
        StateS = State.^2;
%         StateE = exp(State);
        fnameS = sprintf('Sstate%d.mat', stub);
        save(fnameS, 'StateS');
%         fnameE = sprintf('Estate%d.mat', stub);
%         save(fnameE, 'StateE');
        fileIDS = fopen(sprintf('Sstate%d.txt', stub),'w');
        fprintf(fileIDS, '----t=%d----.\n', t);
        fprintf(fileIDS, '%d\n', StateS);
        fclose(fileIDS);
%         fileIDE = fopen(sprintf('Estate%d.txt', stub),'w');
%         fprintf(fileIDE, '----t=%d----.\n', t);
%         fprintf(fileIDE, '%d\n', StateE);
%         fclose(fileIDE);
        stub = stub + 0.1*NetSize;
        
    else if infCnt > stub+100
          fprintf('Time Step too Coarse: Reset Infection Rate!\n')
          break
        end
    end
end
fprintf('Time runs out. Current Infection Count = %d\n', sum((State>0)));
