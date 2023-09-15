% SI Epidemics on Aw
% Configuration Parameters
% Initialization



iter_signal=1000;
for iiiiii=1:iter_signal
load('AdjMw.mat');
NetSize = size(Aw,2);
betaMax = 0.5; % infection rate

tMax = 5000; %max # of time steps
I0 = 1; % No. of seeds

% Settings
Beta = betaMax ./ Aw; % Assume infection rate to be inverse of distance
Beta(Beta==inf) = 0;
stub = 100;
MaxRuntime = 10;

    
    % Initialization
    Count = zeros(1,tMax);
  fprintf('-----------Initialilzation----------\n');
InitState = zeros(1,NetSize);
if I0 == 1
k = randi(NetSize);
InitState(k) = 1;
fprintf('Seed # 1, Node %i\n', k);
else
curI = 0;
load('DistSF1.mat')
[sortV, sortI] = sort(Dist(:), 'descend');
for i = 1:2:NetSize % two seeds every time
if curI == I0
break
end
Ind = sortI(i);
[x,y] = ind2sub(size(Dist),Ind);
if InitState(x) == 0;
InitState(x) = 1;
curI = curI+1;
fprintf('Seed # %i, Node %i\n', curI, x);
if curI == I0
break
end
if InitState(y) == 0;
InitState(y) = 1;
curI = curI+1;
fprintf('Seed # %i, Node %i\n', curI, y);
end
end
end
end
fprintf('---------Initialilzation Ends--------\n');
    
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
        fnameS = sprintf('Sstate_%d_%d.mat',iiiiii, stub);
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
end







% % % % % % % % % % % %% Processing
% % % % % % % % % % % 
% % % % % % % % % % % p_false_positive=.1;
% % % % % % % % % % % A = importdata('AdjMw.mat');
% % % % % % % % % % % size=size(A,1);
% % % % % % % % % % % D = zeros(size);
% % % % % % % % % % % alpha = 0.2; % tail control variable
% % % % % % % % % % % gamma = 0.5; % head control variable
% % % % % % % % % % % for i=1:size
% % % % % % % % % % %     D(i,i)=sum(A(i,:));
% % % % % % % % % % % end
% % % % % % % % % % % L=D-A;
% % % % % % % % % % % [EigVector,EigValue] = eig(L);
% % % % % % % % % % % 
% % % % % % % % % % % % Signal
% % % % % % % % % % % ss=zeros(9,size);
% % % % % % % % % % % answer = ss; % FT of signal (w/ negative values)
% % % % % % % % % % % smoothness_signal = zeros(9,1); %smoothness
% % % % % % % % % % % signalEng = ss; % Energy
% % % % % % % % % % % EngHR_s = zeros(1,9);
% % % % % % % % % % % EnglR_s = zeros(1,9);
% % % % % % % % % % % 
% % % % % % % % % % % for dd=100:100:900
% % % % % % % % % % %     number_of_false_positives=(size-dd)*p_false_positive;
% % % % % % % % % % %     false_positive_indices= randperm(length(find(ss(dd/100,:)==0)));
% % % % % % % % % % %     myfilename = sprintf('Trace3_Square/Trace3_Rep1/Sstate%d.mat', dd);
% % % % % % % % % % % 
% % % % % % % % % % %     qqq=importdata(myfilename);
% % % % % % % % % % %     ss(dd/100,:) =qqq;
% % % % % % % % % % %     ss(ss(dd/100,:)==0) = -max(ss(dd/100,:));
% % % % % % % % % % %     smooth_square_signal=0;
% % % % % % % % % % %     for i=1:size
% % % % % % % % % % %         FT_s(i,:)=dot(ss(dd/100,:),EigVector(:,i));
% % % % % % % % % % %         for j=1:size
% % % % % % % % % % %             smooth_square_signal=smooth_square_signal+ A(i,j)*(ss(dd/100,i)-ss(dd/100,j))^2;
% % % % % % % % % % %         end
% % % % % % % % % % %     end
% % % % % % % % % % %     answer(dd/100,:)=FT_s';
% % % % % % % % % % %     for i=1:size
% % % % % % % % % % %         if i ==1
% % % % % % % % % % %             signalEng(dd/100,i) = 0;
% % % % % % % % % % %         else
% % % % % % % % % % %             signalEng(dd/100,i)=signalEng(dd/100,i-1)+abs(answer(dd/100,i));
% % % % % % % % % % %         end
% % % % % % % % % % %     end
% % % % % % % % % % %     smoothness_signal(dd/100)=sqrt(smooth_square_signal);
% % % % % % % % % % %     EngHR_s(dd/100) = sum(abs(answer(dd/100, (1-alpha)*size:end))) / sum(abs(answer(dd/100, :)));
% % % % % % % % % % %     EngLR_s(dd/100) = sum(abs(answer(dd/100, 1:gamma*size))) / sum(abs(answer(dd/100, :)));
% % % % % % % % % % % end
