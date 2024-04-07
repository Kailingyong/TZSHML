function [LTrain1,LTrain2,LTrain3,LTrain4] = Noiselabel_function_3(LTrain)
% rand('seed',1);
%%
LTrain1 = LTrain;
LTrain2 = LTrain;
LTrain3 = LTrain;
LTrain4 = LTrain;

%%
row = size(LTrain,1);
Rnum4 = round( 0.6 * row);
Rnum3 = round(0.4 * row);
Rnum2 = round(0.2 * row);



%%
Rdata4 = randperm(row,Rnum4); %���ѡ��ratio*n(nΪ��ǩ������1�ĸ�����ratioΪѡ�����)������Ϊȱʧ��ǩ
Rdata4 = sort(Rdata4);
Rdata3 = Rdata4(1:Rnum3);
Rdata2 = Rdata3(1:Rnum2);




for i = 1:Rnum2
    for j = 1:size(LTrain,2)
        if LTrain(Rdata2(i),j)==1
           LTrain2(Rdata2(i),j) = 0;
        else
            LTrain2(Rdata2(i),j) = 1;
        end
    end
end
  
for i = 1:Rnum3
    for j = 1:size(LTrain,2)
        if LTrain(Rdata3(i),j)==1
           LTrain3(Rdata3(i),j) = 0;
        else
            LTrain3(Rdata3(i),j) = 1;
        end
    end
end

for i = 1:Rnum4
    for j = 1:size(LTrain,2)
        if LTrain(Rdata4(i),j)==1
           LTrain4(Rdata4(i),j) = 0;
        else
            LTrain4(Rdata4(i),j) = 1;
        end
    end
end

end

