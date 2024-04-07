function [B, P1, P2, t1, t2] = TZSHML(param, A, X1_UR, X1_UQ, X2_UR, X2_UQ, L_UR, L_UQ, L, X1_l, X2_l, Y_l, X1_m, X2_m, Y_m)



%% 参数
%% Stage1
alphe1 = param.alphe1; alphe2 = param.alphe2;  alphe3 = param.alphe3; alphe4 = param.alphe4; lambda = param.lambda;

%% Stage2
alphe5 = param.alphe5; alphe6 = param.alphe6;  alphe7 = param.alphe7; sf = param.sf;

%% Stage3
alphe8 = param.alphe8; alphe9 = param.alphe9; theta = param.theta;

bit = param.nbits;
r = 400;
kdim = bit/sf;

[~, c] = size(Y_l);
etaX = 0.5; etaY = 0.5;
beta1 = 0.5; beta2 = 0.2;  beta3 = 0.3;


%%   Stage1 标签恢复学习
if exist('X1_m','var')
%%
 [~, dx11] = size(X1_l); [~ , dx21] = size(X2_l); 



Xl = X1_l';  Yl = X2_l';  Y_l = Y_l';
%%  
U = (NormalizeFea(Y_l',1))';
Ll_=(NormalizeFea(Y_l',1))';

W1 = rand(r,dx11);
R1 = rand(c,r);
W2 = rand(r,dx21);
R2 = rand(c,r);




for iter = 1:param.iter
    U = (2 * alphe2 * eye(c)  + alphe1 * R1 * W1 * Xl * Xl' * W1' * R1' + alphe1 * R2 * W2 * Yl * Yl' * W2' * R2')...
        \ (alphe2 * R1 * W1 * Xl + alphe2 * R2 * W2 * Yl + alphe1 * R1 * W1 * Xl * Ll_'* Ll_ + alphe1 * R2 * W2 * Yl * Ll_'* Ll_);

    
RW1 = R1*W1;
RW2 = R2*W2;
v1  = sqrt(sum(RW1.*RW1,2)+eps);
v2  = sqrt(sum(RW2.*RW2,2)+eps);
M1  = diag(1./(2*v1));
M2  = diag(1./(2*v2));    
    
    
%% W1 W2
WA1 = pinv(alphe1 * R1' * U * U' * R1 + alphe2 * R1' * R1 )*(lambda * R1' * M1 * R1);
WB1 = Xl * Xl';
WC1 = pinv(alphe1 * R1' * U * U' * R1 + alphe2 * R1' * R1 )*(alphe1 * R1' * U * Ll_' * Ll_ * Xl' + alphe2 * R1' * U * Xl');
W1 = sylvester((WA1),(WB1),(WC1));

WA2 = pinv(alphe1 * R2' * U * U' * R2 + alphe2 * R2' * R2 )*(lambda * R2' * M2 * R2);
WB2 = Yl * Yl';
WC2 = pinv(alphe1 * R2' * U * U' * R2 + alphe2 * R2' * R2 )*(alphe1 * R2' * U * Ll_' * Ll_ * Yl' + alphe2 * R2' * U * Yl');
W2 = sylvester((WA2),(WB2),(WC2));


%% R1 R2
RA1 = pinv(alphe1 * U * U' + alphe2 * eye(c))*(lambda * M1);
RB1 = (W1 * Xl * Xl'* W1')*pinv(W1 * W1');
RC1 = pinv(alphe1 * U * U' + alphe2 * eye(c))*(alphe1 * U * Ll_' * Ll_ * Xl'* W1' + alphe2 * U * Xl'* W1')*pinv(W1 * W1');
R1 = sylvester((RA1),(RB1),(RC1));

RA2 = pinv(alphe1 * U * U' + alphe2 * eye(c))*(lambda * M2);
RB2 = (W2 * Yl * Yl'* W2')*pinv(W2 * W2');
RC2 = pinv(alphe1 * U * U' + alphe2 * eye(c))*(alphe1 * U * Ll_' * Ll_ * Yl'* W2' + alphe2 * U * Yl'* W2')*pinv(W2 * W2');
R2 = sylvester((RA2),(RB2),(RC2));
    
end

Yu = 0.5 * R1 * W1 * X1_m' + 0.5 * R2 * W2 * X2_m';
Yu (Yu < 0) = 0;
Ym = Yu;
Yn = Yu;   Y = Y_m';
[~, n] = size(X1_m'); 
%% 计算 D
inx = (Y == -1);
D = ones(c,n);
D(inx) = 0;

%% Calculate Yn
manifold1 = euclideanfactory(c, n);
problem1.M = manifold1;
problem1.cost  = @(Yn) alphe3*norm(D.*(Y-Yn), 'fro') ^ 2 +  alphe4*norm(Yn-Ym, 'fro') ^ 2;
problem1.egrad = @(Yn) alphe3*2*(D.*(Yn-Y)) + alphe4*2*(Yn-Ym);


options.maxtime = 10;
[Yn, ~, ~, ~] = trustregions(problem1, [], options);
inx = (Yn <= 0.5);
Yn(inx) = 0;
inx = (Yn > 0.5);
Yn(inx) = 1;


%% 先恢复缺失标签，再拼接正常标签
X1 = [X1_l;X1_m]';  X2 = [X2_l;X2_m]'; 
Yn = [Y_l,Yn];
[~, n] = size(X1); 



else
    X1 = X1_l';  X2 = X2_l';  Yn = Y_l';
end

%% Calculate E
for i= 1:size(Yn,2)
    E(i,:) = (A*Yn(:,i))./(sum(Yn(:,i))+eps);
end


%%   Stage2 哈希码学习和哈希函数学习
%%
[dx1,n] = size(X1); 
    
%% 初始化相关参数
Z = randn(size(A,1), size(A,1));
V = sqrt(n*bit/kdim)*orth(rand(n,kdim));
B = rsign(V,bit,kdim);
W = rand(size(A,1),size(B,2));

 %% 初始化相关参数
t1 = ones(1,kdim);
t2 = ones(1,kdim);
en = ones(n,1);
P1 = rand(dx1,kdim);
P2 = rand(dx1,kdim);


X1X1 = X1* X1';
X2X2 = X2* X2'; 


LSet = cell(1,size(param.km,2));
for j = 1:size(param.km,2)
    [a,~] = kmeans(Yn',param.km(j),'Distance','sqEuclidean');
    LSet{j} = sparse(1:size(Yn,2),a,1);
    LSet{j} = full(LSet{j});
end

 for iter = 1:param.iter 

    %% update B
    BCluster = zeros(size(B));
    for j = 1: size(param.km,2)
        BCluster = BCluster + param.km(j)/sum(param.km)*LSet{j}*(LSet{j}'*V);
    end
    B = rsign(alphe7*V + alphe6*bit* (beta1*Yn'*(Yn*V) + beta2*etaX*X1'*(X1*V) +beta2*etaY*X2'*(X2*V) +beta3*BCluster)...
            +theta*n*bit/kdim*ones(n,kdim),bit,kdim); %bit balance (extra)
     
    %% update V
    BCluster = zeros(size(V));
    for j = 1: size(param.km,2)
        BCluster = BCluster + param.km(j)/sum(param.km)*LSet{j}*(LSet{j}'*B);
    end
    V11 = alphe5*E*W +alphe9*(X1'*P1+X2'*P2+en*t1+en*t2)+alphe6*bit* (beta1*Yn'*(Yn*B) + beta2*etaX*X1'*(X1*B) +beta2*etaY*X2'*(X2*B) +beta3*BCluster);
    V = V11*pinv((alphe5+alphe7+alphe9)*eye(kdim) + alphe6*B'*B);
            
    %% update W
    W = pinv(alphe5*E'*E+lambda * eye(size(E,2)))* (alphe5*E'*V);

    %% update P1 P2
    P1 = pinv((alphe8+alphe9)*X1X1+lambda*eye(dx1))*(alphe8*X1*en*t1+alphe8*X1*X2'*P2-alphe8*X1*en*t2+alphe9*X1*V-alphe9*X1*en*t1);
    P2 = pinv((alphe8+alphe9)*X2X2+lambda*eye(dx1))*(alphe8*X2*en*t2+alphe8*X2*X1'*P1-alphe8*X2*en*t1+alphe9*X2*V-alphe9*X2*en*t2);
    
    %% update t1 t2
     t1 = (1/n)*en'*(alphe9*V-alphe9*X1'*P1+alphe8*X1'*P1-alphe8*X2'*P2+alphe8*en*t2);
     t2 = (1/n)*en'*(alphe9*V-alphe9*X2'*P2+alphe8*X2'*P2-alphe8*X1'*P1+alphe8*en*t1);
   
 end

end 

