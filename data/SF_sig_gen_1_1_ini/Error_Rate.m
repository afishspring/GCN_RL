clc
clear
Smth_noise=load('CI_sm.mat')
Smth_noise=cell2mat(struct2cell(Smth_noise));
ECLow_noise=load('CI_ecl.mat')
ECLow_noise=cell2mat(struct2cell(ECLow_noise));
ECHigh_noise=load('CI_ech.mat')
ECHigh_noise=cell2mat(struct2cell(ECHigh_noise));
A = importdata('AdjMw.mat');
size=size(A,1);
D = zeros(size);
iter=1000
alpha = 0.3; % tail control variable
gamma = 0.3; % head control variable
for i=1:size
    D(i,i)=sum(A(i,:));
end
L=D-A;
[EigVector,EigValue] = eig(L);
smoothness_signal = zeros(iter,9); %smoothness
EngHR_s = zeros(iter,9);
EngLR_s = zeros(iter,9);
%%
for iiii=1:iter
p_false_positive=0;
% Signal
ss=zeros(9,size);
answer = ss; % FT of signal (w/ negative values)
  
signalEng = ss; % Energy
  
  
for dd=100:100:900
     
    [iiii,dd]
    number_of_false_positives=(size-dd)*p_false_positive;
    false_positive_indices= randperm(length(find(ss(dd/100,:)==0)));
%     FPFNSstate_SF_1_1_ini_%d_%d.mat
    myfilename = sprintf('FPFNSstate_SF_1_1_ini_%d_%d.mat',iiii, dd);
  
    qqq=importdata(myfilename);
    ss(dd/100,:) =qqq;
    ss(ss(dd/100,:)~=0) = 1000;
%     ss(ss(dd/100,:)==0) = -max(ss(dd/100,:));
    ss(ss(dd/100,:)==0) = -1000;
    smooth_square_signal=0;
    for i=1:size
        FT_s(i,:)=dot(ss(dd/100,:),EigVector(:,i));
        for j=1:size
            smooth_square_signal=smooth_square_signal+ A(i,j)*(ss(dd/100,i)-ss(dd/100,j))^2;
        end
    end
    answer(dd/100,:)=FT_s';
    for i=1:size
        if i ==1
            signalEng(dd/100,i) = 0;
        else
            signalEng(dd/100,i)=signalEng(dd/100,i-1)+abs(answer(dd/100,i));
        end
    end
    smoothness_signal(iiii,dd/100)=sqrt(smooth_square_signal);
    EngHR_s(iiii,dd/100) = sum(abs(answer(dd/100, (1-alpha)*size:end))) / sum(abs(answer(dd/100, :)));
    EngLR_s(iiii,dd/100) = sum(abs(answer(dd/100, 1:gamma*size))) / sum(abs(answer(dd/100, :)));
end
  
end
%% Smoothness
iter=iter;
% % % % figure
Error=zeros(9,1);
for i=1:iter
    for d=1:9
        i
   if(Smth_noise(2,d) >smoothness_signal(i,d) && smoothness_signal(i,d) > Smth_noise(1,d))
       Error(d)=Error(d)+1;
   end
    end
end
Error=Error./iter;
Y=[1-Error(1) 1-Error(2) 1-Error(3) 1-Error(4) 1-Error(5) 1-Error(6) 1-Error(7) 1-Error(8) 1-Error(9)]
X=[100 200 300 400 500 600 700 800 900]
plot(X,Y)
xlabel('Number of infected nodes')
ylabel('Probability of detection using the Smoothness metric')
title('SF graph type 1 vs 1 initial infected nodes')
  
  
% Energy CL
figure
ErrorELow=zeros(9,1);
for i=1:iter
    for d=1:9
   if(ECLow_noise(2,d) >EngLR_s(i,d) && EngLR_s(i,d) > ECLow_noise(1,d))
       [ECLow_noise(2,d),ECLow_noise(1,d),EngLR_s(i,d)]
       [i,d]
       ErrorELow(d)=ErrorELow(d)+1;
   end
    end
end
ErrorELow=ErrorELow/iter;
Y=[1-ErrorELow(1) 1-ErrorELow(2) 1-ErrorELow(3) 1-ErrorELow(4) 1-ErrorELow(5) 1-ErrorELow(6) 1-ErrorELow(7) 1-ErrorELow(8) 1-ErrorELow(9)]
X=[100 200 300 400 500 600 700 800 900]
plot(X,Y)
xlabel('Number of infected nodes')
ylabel('Probability of detection using the  ECRL')
title('SF graph type 1 vs 1 initial infected nodes')
  
% Energy CH
figure
ErrorEH=zeros(9,1);
for i=1:iter
    for d=1:9
        i
   if(ECHigh_noise(2,d) >EngHR_s(i,d) && EngHR_s(i,d) > ECHigh_noise(1,d))
       ErrorEH(d)=ErrorEH(d)+1;
   end
    end
end
ErrorEH=ErrorEH/iter;
Y=[1-ErrorEH(1) 1-ErrorEH(2) 1-ErrorEH(3) 1-ErrorEH(4) 1-ErrorEH(5) 1-ErrorEH(6) 1-ErrorEH(7) 1-ErrorEH(8) 1-ErrorEH(9)]
X=[100 200 300 400 500 600 700 800 900]
plot(X,Y)
xlabel('Number of infected nodes')
ylabel('Probability of detection using the  ECRH')
title('SF graph type 1 vs 1 initial infected nodes')