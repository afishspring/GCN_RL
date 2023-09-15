clear; close all;
load('DistER75.mat')

% Initialize logs (Infection ratio ranges from 0.1 to 0.9)
MaxRuntime = 1000;
SucEpiBall = zeros(MaxRuntime,9);
%SucRanBall = zeros(MaxRuntime,9);
SucEpiDen = zeros(MaxRuntime,9);
%SucRanDen = zeros(MaxRuntime,9);

% Epidemics Generation
NetType = 'ER_75';
NoSeeds = 1;
NetSize = size(Dist,2);

fileID1 = fopen('Det_Ball_Epi.txt','w');
fprintf(fileID1, 'Network = %s, NoSeeds = %i, Size = 1000, runTimes = %i, Method = Ball Density.\r\n', NetType, NoSeeds, MaxRuntime);
fprintf(fileID1, '----------------------------------------------------------------------------------\r\n');
fprintf(fileID1, '%6s %6s %6s %6s %6s %6s %6s %6s %6s\r\n', '10%','20%','30%','40%','50%','60%','70%','80%','90%');

% fileID2 = fopen('Det_Ball_Ran.txt','w');
% fprintf(fileID2, 'Network = ER75, Size = 1000, runTimes = %i, Method = Ball Density.\r\n', MaxRuntime);
% fprintf(fileID2, '----------------------------------------------------------------------------------\r\n');
% fprintf(fileID2, '%6s %6s %6s %6s %6s %6s %6s %6s %6s\r\n', '10%','20%','30%','40%','50%','60%','70%','80%','90%');

fileID3 = fopen('Det_Den_Epi.txt','w');
fprintf(fileID3, 'Network = %s, NoSeeds = %i, Size = 1000, runTimes = %i, Method = Relative Ball Density.\r\n', NetType, NoSeeds, MaxRuntime);
fprintf(fileID3, '----------------------------------------------------------------------------------\r\n');
fprintf(fileID3, '%6s %6s %6s %6s %6s %6s %6s %6s %6s\r\n', '10%','20%','30%','40%','50%','60%','70%','80%','90%');

% fileID4 = fopen('Det_Den_Ran.txt','w');
% fprintf(fileID4, 'Network = ER75, Size = 1000, runTimes = %i, Method = Relavtive Ball Density.\r\n', MaxRuntime);
% fprintf(fileID4, '----------------------------------------------------------------------------------\r\n');
% fprintf(fileID4, '%6s %6s %6s %6s %6s %6s %6s %6s %6s\r\n', '10%','20%','30%','40%','50%','60%','70%','80%','90%');


% Outer loop of repeating the experiment.
for run=1:MaxRuntime
%     dirName = sprintf('./Grid%s', int2str(run));
%     oldDir = cd(dirName);
    for j=1:9
        ratio = 0.1+0.1*(j-1);
        number = int16(ratio*NetSize);
        fname = sprintf('FPFNSstate_%s_%s_ini_%s_%s.mat', NetType, int2str(NoSeeds), int2str(run), int2str(number));

        load(fname)
        result1 = Det_Ball(Dist, FPFNStates);
        result3 = Det_Den(Dist, FPFNStates);
        SucEpiBall(run,j) = strcmp(result1, 'Epidemic');
        SucEpiDen(run,j) = strcmp(result3, 'Epidemic');
        fprintf('Running %i-th time for epidemic%s.\n', run, int2str(number));
        
%         noise=zeros(1,NetSize);
%         noise(randperm(numel(noise), number)) = 1;
%         result2 = Det_Ball(Dist, noise);
%         result4 = Det_Den(Dist, noise);
%         SucRanBall(run,j) = strcmp(result2, 'Random Illness');
%         SucRanDen(run,j) = strcmp(result4, 'Random Illness');
%         fprintf('Running %i-th time for noise%s. \n', run, int2str(number));
       
    end
    fprintf(fileID1, '%6d %6d %6d %6d %6d %6d %6d %6d %6d\r\n', round(SucEpiBall(run,:)));
    %fprintf(fileID2, '%6d %6d %6d %6d %6d %6d %6d %6d %6d\r\n', round(SucRanBall(run,:)));
    fprintf(fileID3, '%6d %6d %6d %6d %6d %6d %6d %6d %6d\r\n', round(SucEpiDen(run,:)));
    %fprintf(fileID4, '%6d %6d %6d %6d %6d %6d %6d %6d %6d\r\n', round(SucRanDen(run,:)));
%     cd;
end

fclose(fileID1);
%fclose(fileID2);
fclose(fileID3);
%fclose(fileID4);

figure
plot(0.1:0.1:0.9, mean(SucEpiBall), 'r-', 0.1:0.1:0.9, mean(SucEpiDen), 'r-.')
xlabel('Expected Infection Ratio')
ylabel('Successful Ratio')
legend('EpiBall', 'EpiDen')
grid on