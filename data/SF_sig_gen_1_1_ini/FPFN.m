number=[100 200 300 400 500 600 700 800 900];
FPFNratio=  [10 20 30 40 50 60 70 80 90];
total_iter=1000;
      

for i = 1    : total_iter
     for j=1:length(number)
         [i, number(j)]
          myfilename = sprintf('Sstate_%d_%d.mat',i, number(j));

         qqq=importdata(myfilename);
%         myfilename = sprintf('Grid%d/State%d.mat',i, number(j));
%         qqq=importdata(myfilename);
          States=qqq;
        infected_indices= find(States~=0);
        len_infected_indices=length(infected_indices);
        p1 = randperm(len_infected_indices);
        
        not_infected_indices=find(States==0);
        len_not_infected_indices=length(not_infected_indices);
        p2 = randperm(len_not_infected_indices);
        for k= 1 :  FPFNratio(j)
           
            States(infected_indices (p1(k)))=0;
            States(not_infected_indices(p2(k)))=randi([1 max(States)],1,1);
            
        end
        
        
        FPFNStates=States;

        myfilenameWrite = sprintf('FPFNSstate_SF_1_1_ini_%d_%d.mat',i, number(j));
        save(myfilenameWrite,'FPFNStates')

  
    end

end

