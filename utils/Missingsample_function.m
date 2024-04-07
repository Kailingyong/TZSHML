function [X2_m,X3_m,X4_m,Y2_m,Y3_m,Y4_m,X2_l,X3_l,X4_l,Y2_l,Y3_l,Y4_l,L2_m,L3_m,L4_m,L2_l,L3_l,L4_l,L2,L3,L4] = Missingsample_function(X,Y,LTrain,L_SR)
% % rand('seed',1);
% % %%
% % LTrain1 = LTrain;
% % LTrain2 = LTrain; 
% % LTrain3 = LTrain;
% % LTrain4 = LTrain;

%%  ѡ���ǩȱʧ��������
[n,m] = size(LTrain);
Rnum4 = round(0.6 * n);
Rnum3 = round(0.4 * n);
Rnum2 = round(0.2 * n);

%% �ҵ�ȱʧ�����ͷ�ȱʧ����������
Rdata4 = randperm(n,Rnum4); %���ѡ��ratio*n(nΪ��ǩ������1��0��������ratioΪѡ�����)������Ϊȱʧ��ǩ
Rdata3 = Rdata4(1:Rnum3);
Rdata2 = Rdata3(1:Rnum2);
Rdata4_miss = sort(Rdata4);
Rdata3_miss = sort(Rdata3);
Rdata2_miss = sort(Rdata2);


Rdata4_unmiss = [1:n]';
Rdata3_unmiss = [1:n]';
Rdata2_unmiss = [1:n]';

Rdata4_unmiss(Rdata4_miss) = [];
Rdata3_unmiss(Rdata3_miss) = [];
Rdata2_unmiss(Rdata2_miss) = [];

%%  �����ݼ���Ϊȱʧ�����ͷ�ȱʧ����
X2_m = X(Rdata2_miss,:);
X3_m = X(Rdata3_miss,:);
X4_m = X(Rdata4_miss,:);
Y2_m = Y(Rdata2_miss,:);
Y3_m = Y(Rdata3_miss,:);
Y4_m = Y(Rdata4_miss,:);
L2_m = LTrain(Rdata2_miss,:);
L3_m = LTrain(Rdata3_miss,:);
L4_m = LTrain(Rdata4_miss,:);
L22_m = L_SR(Rdata2_miss,:);
L33_m = L_SR(Rdata3_miss,:);
L44_m = L_SR(Rdata4_miss,:);


X2_l = X(Rdata2_unmiss,:);
X3_l = X(Rdata3_unmiss,:);
X4_l = X(Rdata4_unmiss,:);
Y2_l = Y(Rdata2_unmiss,:);
Y3_l = Y(Rdata3_unmiss,:);
Y4_l = Y(Rdata4_unmiss,:);
L2_l = LTrain(Rdata2_unmiss,:);
L3_l = LTrain(Rdata3_unmiss,:);
L4_l = LTrain(Rdata4_unmiss,:);
L22_l = L_SR(Rdata2_unmiss,:);
L33_l = L_SR(Rdata3_unmiss,:);
L44_l = L_SR(Rdata4_unmiss,:);



L2 = [L22_l;L22_m];
L3 = [L33_l;L33_m];
L4 = [L44_l;L44_m];

%% Ϊȱʧ�������ȱʧ��ǩ
Rm2 = round(0.2 * size(L2_m,1)*size(L2_m,2));
Rm3 = round(0.4 * size(L3_m,1)*size(L3_m,2));
Rm4 = round(0.6 * size(L4_m,1)*size(L4_m,2));

Rlabel2 = randperm(size(L2_m,1)*size(L2_m,2),Rm2); %���ѡ��ratio*n(nΪ��ǩ������1��0��������ratioΪѡ�����)������Ϊȱʧ��ǩ
Rlabel3 = randperm(size(L3_m,1)*size(L3_m,2),Rm3);
Rlabel4 = randperm(size(L4_m,1)*size(L4_m,2),Rm4);

Rlabel2 = sort(Rlabel2);
Rlabel3 = sort(Rlabel3);
Rlabel4 = sort(Rlabel4);


%% ����ǩ������ƽΪһ��������������������
L2_m1 = reshape(L2_m,1,[]);
L3_m1 = reshape(L3_m,1,[]);
L4_m1 = reshape(L4_m,1,[]);


%% ȱʧ��ǩ���
L2_m1(Rlabel2) = -1;
L3_m1(Rlabel3) = -1;
L4_m1(Rlabel4) = -1;


%% �ָ����ȱʧ������ı�ǩ
L2_m = reshape(L2_m1,length(Rdata2_miss),m);
L3_m = reshape(L3_m1,length(Rdata3_miss),m);
L4_m = reshape(L4_m1,length(Rdata4_miss),m);

end

