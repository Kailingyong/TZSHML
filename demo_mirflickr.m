clear;clc

addpath dataset;
addpath utils;
addpath(genpath('Manopt_7.1'));
 %% 数据处理
 %% load data
load './dataset/mirflickr25k_1.mat'
load './dataset/mirflickr_A.mat'
mirflickr_attributes = y3;

I_tr = I_tr1;
T_tr = T_tr1;
L_tr = L_tr1; 

XTrain = I_tr;  YTrain = T_tr;
XTest  = I_te;  YTest  = T_te;
LTrain = L_tr;  LTest  = L_te;

L_tr_Matrix = L_tr;
L_te_Matrix = L_te;

XTest  = bsxfun(@minus, XTest, mean(XTrain, 1)); 
XTrain = bsxfun(@minus, XTrain, mean(XTrain,1));
YTest  = bsxfun(@minus, YTest, mean(YTrain, 1));    
YTrain = bsxfun(@minus, YTrain, mean(YTrain,1));

[XKTrain,XKTest] = Kernelize(XTrain, XTest, 1000) ; [YKTrain,YKTest]=Kernelize(YTrain, YTest, 1000);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));

% nbitset  = [2,4,8,16,32]; 
nbitset  = [2]; 
for bit = 1:length(nbitset)  
run = 10;
for i = 1 : run
    tic 
fprintf('Unseen classes:\n');
load seenClass.mat;
load unseenClass.mat;
 unseenClass
 
k=1;
 k1=1;
for j=1:size(L_tr,1)
    if L_tr(j,unseenClass(:,1))==1 || L_tr(j,unseenClass(:,2))==1|| L_tr(j,unseenClass(:,3))==1|| L_tr(j,unseenClass(:,4))==1|| L_tr(j,unseenClass(:,5))==1
       index_unseen_in_tr(k,:) = j;
       k=k+1;
    else
       index_seen_in_tr(k1,:) = j;
       k1=k1+1; 
    end 
end

 k=1;
 k1=1;
for j=1:size(L_te,1)
    if L_te(j,unseenClass(:,1))==1 || L_te(j,unseenClass(:,2))==1|| L_te(j,unseenClass(:,3))==1|| L_te(j,unseenClass(:,4))==1|| L_te(j,unseenClass(:,5))==1
       index_unseen_in_te(k,:) = j;
       k=k+1;
    else
       index_seen_in_te(k1,:) = j;
       k1=k1+1; 
    end 
end


% train data of seen class. same as retrieal data
X1_SR = XKTrain(index_seen_in_tr,:);
X2_SR = YKTrain(index_seen_in_tr,:);
L_SR = L_tr_Matrix(index_seen_in_tr,:);

X1_SQ = XKTest(index_seen_in_te,:);
X2_SQ = YKTest(index_seen_in_te,:);
L_SQ = L_te_Matrix(index_seen_in_te,:);

% data split of unseen data
X1_UR = XKTrain(index_unseen_in_tr,:);
X2_UR = YKTrain(index_unseen_in_tr,:);
L_UR = L_tr_Matrix(index_unseen_in_tr,:);

X1_UQ = XKTest(index_unseen_in_te,:);
X2_UQ = YKTest(index_unseen_in_te,:);
L_UQ = L_te_Matrix(index_unseen_in_te,:);


S = mirflickr_attributes(seenClass,:);


%% data preprocessing
X = X1_SR;
Y = X2_SR;
L = L_SR(:,seenClass);


%% 对训练集设置标签缺失
[X2_m,X3_m,X4_m,Y2_m,Y3_m,Y4_m,X2_l,X3_l,X4_l,Y2_l,Y3_l,Y4_l,L2_m,L3_m,L4_m,L2_l,L3_l,L4_l,L2,L3,L4] = Missingsample_function(X,Y,L,L_SR);


%% 参数设置
%% Stage1
param.alphe1 = 1e0; param.alphe2 = 1e0;  param.alphe3 = 1e0; param.alphe4 = 1e0; param.lambda = 1e0; param.theta = 1e0;

%% Stage2
param.alphe5 = 1e0; param.alphe6 = 1e0;  param.alphe7 = 1e0; param.sf = 0.05; param.km = [50 100 200]; % MIRFLICKR

%% Stage3
param.alphe8 = 1e0; param.alphe9 = 1e-1; 

param.iter = 5;



eva_info = cell(1,length(nbitset));

param.nbits = nbitset(bit);
kdim = param.nbits/param.sf;
     

 %% 0.2的噪声
 param.nbits = nbitset(bit);
 [B, P1, P2, t1, t2] = TZSHML(param, S', X1_UR, X1_UQ, X2_UR, X2_UQ, L_UR, L_UQ, L2, X2_l, Y2_l, L2_l, X2_m, Y2_m, L2_m); 
 
 
%%
    rBX = [rsign(X1_UR * P1 - ones(size(X1_UR,1),1) * t1,param.nbits,kdim);B];
    qBX = rsign(X1_UQ * P1- ones(size(X1_UQ,1),1) * t1,param.nbits,kdim);
    rBY = [rsign(X2_UR * P2 - ones(size(X2_UR,1),1) * t2,param.nbits,kdim);B];
    qBY = rsign(X2_UQ * P2 - ones(size(X2_UQ,1),1) * t2,param.nbits,kdim);

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L2], L_UQ);
    map2(i,1) = mapTI(end);

    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L2], L_UQ);
    map2(i,2) =  mapIT(end);


 %% 0.4的噪声
   param.nbits = nbitset(bit); 
   [B, P1, P2, t1, t2] = TZSHML(param, S', X1_UR, X1_UQ, X2_UR, X2_UQ, L_UR, L_UQ, L3, X3_l, Y3_l, L3_l, X3_m, Y3_m, L3_m);   
   
   
%%
    rBX = [rsign(X1_UR * P1 - ones(size(X1_UR,1),1) * t1,param.nbits,kdim);B];
    qBX = rsign(X1_UQ * P1- ones(size(X1_UQ,1),1) * t1,param.nbits,kdim);
    rBY = [rsign(X2_UR * P2 - ones(size(X2_UR,1),1) * t2,param.nbits,kdim);B];
    qBY = rsign(X2_UQ * P2 - ones(size(X2_UQ,1),1) * t2,param.nbits,kdim);

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L3], L_UQ);
    map3(i,1) = mapTI(end);

    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L3], L_UQ);
    map3(i,2) =  mapIT(end);

 %% 0.6的噪声
   param.nbits = nbitset(bit);
 [B, P1, P2, t1, t2] = TZSHML(param, S', X1_UR, X1_UQ, X2_UR, X2_UQ, L_UR, L_UQ, L4, X4_l, Y4_l, L4_l, X4_m, Y4_m, L4_m);
 
 
%%
    rBX = [rsign(X1_UR * P1 - ones(size(X1_UR,1),1) * t1,param.nbits,kdim);B];
    qBX = rsign(X1_UQ * P1- ones(size(X1_UQ,1),1) * t1,param.nbits,kdim);
    rBY = [rsign(X2_UR * P2 - ones(size(X2_UR,1),1) * t2,param.nbits,kdim);B];
    qBY = rsign(X2_UQ * P2 - ones(size(X2_UQ,1),1) * t2,param.nbits,kdim);

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L4], L_UQ);
    map4(i,1) = mapTI(end);

    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L4], L_UQ);
    map4(i,2) =  mapIT(end);


end

map222(bit,:) = mean(map2);
map333(bit,:) = mean(map3);
map444(bit,:) = mean(map4);

end
