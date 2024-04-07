function [LTrain1,LTrain2,LTrain3,LTrain4] = Missinglabel_function(LTrain)
rand('seed',1);
% % %%
% % LTrain1 = LTrain;
% % LTrain2 = LTrain;
% % LTrain3 = LTrain;
% % LTrain4 = LTrain;

%%
[m,n] = size(LTrain);
N = m*n;
Rnum4 = round(0.6 * N);
Rnum3 = round(0.4 * N);
Rnum2 = round(0.2 * N);



%%
Rdata4 = randperm(N,Rnum4); %���ѡ��ratio*n(nΪ��ǩ������1��0��������ratioΪѡ�����)������Ϊȱʧ��ǩ
Rdata3 = Rdata4(1:Rnum3);
Rdata2 = Rdata3(1:Rnum2);
Rdata4 = sort(Rdata4);
Rdata3 = sort(Rdata3);
Rdata2 = sort(Rdata2);

%% ����ǩ������ƽΪһ��������������������
LTrain1 = reshape(LTrain,1,N);
LTrain2 = reshape(LTrain,1,N);
LTrain3 = reshape(LTrain,1,N);
LTrain4 = reshape(LTrain,1,N);

%% ȱʧ��ǩ���
LTrain2(Rdata2) = -1;
LTrain3(Rdata3) = -1;
LTrain4(Rdata4) = -1;


%% �ָ����ȱʧ������ı�ǩ
LTrain1 = reshape(LTrain1,m,n);
LTrain2 = reshape(LTrain2,m,n);
LTrain3 = reshape(LTrain3,m,n);
LTrain4 = reshape(LTrain4,m,n);

end

