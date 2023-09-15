size=1000;
p_false_positive=.1;
A = importdata('AdjMw.mat');
D = zeros(size);
alpha = 0.2; % tail control variable
gamma = 0.5; % head control variable
for i=1:size
    D(i,i)=sum(A(i,:));
end
L=D-A;
[EigVector,EigValue] = eig(L);

% Signal
ss=zeros(9,size);
answer = ss; % FT of signal (w/ negative values)
smoothness_signal = zeros(9,1); %smoothness
signalEng = ss; % Energy
EngHR_s = zeros(1,9);
EnglR_s = zeros(1,9);

for dd=100:100:900
    number_of_false_positives=(size-dd)*p_false_positive;
    false_positive_indices= randperm(length(find(ss(dd/100,:)==0)));
    myfilename = sprintf('Trace5_Multi_Square/Trace5_Rep4_dist2/Sstate%d.mat', dd);

    qqq=importdata(myfilename);
    ss(dd/100,:) =qqq;
    ss(ss(dd/100,:)==0) = -max(ss(dd/100,:));
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
    smoothness_signal(dd/100)=sqrt(smooth_square_signal);
    EngHR_s(dd/100) = sum(abs(answer(dd/100, (1-alpha)*size:end))) / sum(abs(answer(dd/100, :)));
    EngLR_s(dd/100) = sum(abs(answer(dd/100, 1:gamma*size))) / sum(abs(answer(dd/100, :)));
end

% figure
% plot([2:1:1000],abs(signalEng(:,[2:1000])))
% legend('I=100','I=200','I=300','I=400','I=500','I=600','I=700','I=800','I=900')
% title('Cumulative Signal Energy.')

% Noise
iteration=100;
%finalans=zeros(iteration,9,1000);
answern=zeros(1,size);
smoothness_noise=zeros(9,iteration);
final_smooth_noise=zeros(1,9);
EngHR_n = zeros(1,9);
EngLR_n = zeros(1,9);
y1pos=zeros(1,9);
y1neg=zeros(1,9);
y2pos=zeros(1,9);
y2neg=zeros(1,9);
sypos=zeros(1,9);
syneg=zeros(1,9);
for dd=100:100:900 % Outer cycle, I =100 to 900
    Ratio=zeros(1,iteration);
    for iter=1:iteration % Iteration over noise
        fprintf('I = %d, iter=%d\n', dd, iter);
        indices=zeros(1,dd);
        noise=zeros(1,1000);
        noise(randperm(numel(noise), dd)) = randi([0 max(ss(dd/100,:))],dd,1);
        noise(find(noise==0))= - max(ss(dd/100,:));

        smooth_square_noise=0;
        for i=1:size
            FT_n(i,:)=dot(noise,EigVector(:,i));
            for j=1:size
                smooth_square_noise=smooth_square_noise+ A(i,j)*(noise(i)-noise(j))^2;
            end
    %         if i ==1
    %             noiseEng(dd/100,i) = 0;
    %         else
    %             noiseEng(dd/100,i)=noiseEng(dd/100,i-1)+abs(answer(dd/100,i));
    %         end
        end
        answern=FT_n'; % FT of noise at I=dd
        smoothness_noise(dd/100,iter)=sqrt(smooth_square_noise);
        Ratio1(iter) = sum(abs(answern((1-alpha)*size:end))) / sum(abs(answern(:)));
        Ratio2(iter) = sum(abs(answern(1:gamma*size))) / sum(abs(answern(:)));
    end
    EngHR_n(dd/100) = mean(Ratio1);
    SEM = std(Ratio1)/sqrt(length(Ratio1));
    ts = tinv([0.001 0.999], length(Ratio1)-1);
    % CI = mean(x1) + ts*SEM;
    y1neg(dd/100) = abs(ts(1)*SEM);
    y1pos(dd/100) = abs(ts(2)*SEM);
    
    EngLR_n(dd/100) = mean(Ratio2);
    SEM = std(Ratio1)/sqrt(length(Ratio2));
    ts = tinv([0.001 0.999], length(Ratio2)-1);
    % CI = mean(x1) + ts*SEM;
    y2neg(dd/100) = abs(ts(1)*SEM);
    y2pos(dd/100) = abs(ts(2)*SEM);
%     yneg(dd/100) = max(Ratio - mean(Ratio));
%     ypos(dd/100) = max(mean(Ratio) - Ratio);
    final_smooth_noise(dd/100) = mean(smoothness_noise(dd/100,:));
    x2 = smoothness_noise(dd/100,:);
    SEM2 = std(x2)/sqrt(length(x2));
    ts2 = tinv([0.001 0.999], length(x2)-1);
     syneg(dd/100) = abs(ts2(1)*SEM2);
     sypos(dd/100) = abs(ts2(2)*SEM2);
%     syneg(dd/100) = max(smoothness_noise(dd/100,:) - final_smooth_noise(dd/100));
%     sypos(dd/100) = max(final_smooth_noise(dd/100) - smoothness_noise(dd/100,:));
end

figure
plot([100:100:900],smoothness_signal, 'bl')
hold on;
errorbar([100:100:900],final_smooth_noise, syneg, sypos, 'r-o')
legend('signal', 'noise')
title('Smoothness')

figure
plot([100:100:900], EngHR_s, 'bl')
hold on
errorbar([100:100:900], EngHR_n, y1neg, y1pos, 'r-o')
legend('signal', 'noise')
title(['H Energy Concentration Ratio with \alpha = ', num2str(alpha), '.'])

figure
plot([100:100:900], EngLR_s, 'bl')
hold on
errorbar([100:100:900], EngLR_n, y2neg, y2pos, 'r-o')
legend('signal', 'noise')
title(['L Energy Concentration Ratio with \gamma = ', num2str(gamma), '.'])

% Spectrum
% InfCnt=700;
% k = InfCnt/100;
% Freq = diag(EigValue);
% figure
% plot(Freq,answer(k,:), 'bl', Freq, final(k,:), 'r')
% legend('signal', 'noise')
% title(['Spectrum at infection count = ', num2str(InfCnt), '.'])

% Filtered
% alpha = 0.2; % contral variable
% EngHR_s = zeros(1,9);
% EngHR_n = zeros(1,9);
% for k = 1:9
% EngHR_s(k) = sum(abs(answer(k, (1-alpha)*size:end))) / sum(abs(answer(k, :)));
% EngHR_n(k) = sum(abs(final(k, (1-alpha)*size:end))) / sum(abs(final(k, :)));
% end
% figure
% plot([1:9], EngHR_s, 'bl', [1:9], EngHR_n, 'r')
% legend('signal', 'noise')
% title(['Energy Concentration Ratio with \alpha = ', num2str(alpha), '.'])
