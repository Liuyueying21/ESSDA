function [P,U,F,G,Z,M] = ESSDA(Xs,Xs_label,Xt_tl,Xt_tlabel,Xt_ul,Xt_ulabel,alpha,gamma,omega)

% ----------------------------------------------
%               Setting
% ----------------------------------------------
[d1,ns] = size(Xs);
ntl = size(Xt_tl,2);
ntu = size(Xt_ul,2);
nt=ntl+ntu;
nl=ns+ntl;
Xl=[Xs,Xt_tl]; 
class=length(unique(Xt_tlabel));
Xt=[Xt_tl Xt_ul];
X=[Xs,Xt]; 
Max_iter = 20;
Xl_label=[Xs_label;Xt_tlabel];
Xl_c = cell(class,1);
U1 = cell(class,1);
for k = 1:class
    Xl_c{k} = Xl(:,Xl_label==k);   
end
Yl=Construct_Y(Xl_label,nl,class);
n = ns+nt;
F=zeros(class,n);
F(:,1:nl)=Yl;
B = Construct_B(Yl); 
M = ones(class,nl);
anchor_num=2;
m=class*2^anchor_num; 
v = repelem(1:class, 2^anchor_num);
G=Construct_Y(v,m,class);
kk=5;   
I1=eye(class,class);  
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M0 = e*e'*ns*nt/class;
MC=0;
if ~isempty(Xt_tlabel) && length(Xt_tlabel)==ntl
    for c =1:class
        e = zeros(nl,1);
        Ps = length(find(Xs_label==c)) / length(Xs_label);
        Pt = length(find(Xt_tlabel == c)) / length(Xt_tlabel);
        a = Pt / Ps;
        e(Xs_label==c) = 1 / length(find(Xs_label==c));
        e(ns+find(Xt_tlabel==c)) = -a / length(find(Xt_tlabel==c));
        e(isinf(e)) = 0;
        MC = MC + e*e';
    end
end
MC = MC*ns*ntl/(class*class);
% ------------------------------------------------
%                   Main Loop
% ------------------------------------------------
for iter = 1:Max_iter

    % updating P:
    % L_P=||P'Xl-(Yl+B.*M)||_F^2 + ω \sum_n\sum_m ||P'x_i-u_j||_2^2*Z_ij + γ||P||_F^2 + α Tr( P'XM0X'P + P'XlMCXl'P )
    if (iter == 1)
        options = [];
        options.ReducedDim = class;
        [P,~] = PCA1(Xs', options);
    else
        R_1 = zeros(d1,class);
        [row, col] = find(Z);
        for l=1:length(row)
            i=row(l);
            j=col(l);
            R_1 = R_1 + X(:,i)*U(:,j)'*Z(i,j);
        end
        V1 = Yl+B.*M;
        P=(alpha*(X*M0*X'+Xl*MC*Xl')+Xl*Xl'+omega*X*X'+gamma*eye(d1))\(Xl*V1'+omega*R_1);
    end

    % updating M
    %L=||P'Xl-(Yl+B.*M)||_F^2       
    C = P'*Xl-Yl;
    gp = B.*C;
    [numm1,numm2] = size(gp);
    for jk1 = 1:numm1
        for jk2 = 1:numm2
            M(jk1,jk2) = max(gp(jk1,jk2),0);
        end
    end

    % updating U: L_U= ω \sum n\sum m ||P^Tx_i-u_j||^2_2*Z_ij
    if (iter == 1)
        U = [];
        for i=1:class
            U1{i}=P'*Xl_c{i};
            nc=size(U1{i},2);
            [~,U1{i}] = hKM(U1{i}, [1:nc], anchor_num, 1);   
            U = [U U1{i}];
        end
    else
        theta=sum(Z);
        for j=1:m
            sup1 = zeros(d1,1);
            U_row = find(Z(:,j));
            for i=1:length(U_row)
                sup1=sup1+Z(U_row(i),j)*X(:,U_row(i));
            end
            U(:,j) = (P'*sup1)/theta(j);
        end
    end

    % updating G:
    % L_G = \sum n\sum m Z_ij * ||f_i-g_j||^2_2
    % s.t. G\in Ind
    if (iter == 1)
        G=G;
    else
        opt_G=zeros(class,m);
        for i=1:class
            for j=1:m
                col_Z = find(Z(:,j));
                for k=1:length(col_Z)
                    opt_G(i,j)=opt_G(i,j)+Z(col_Z(k),j)*norm(F(:,col_Z(k))-I1(:,i))^2;
                end
            end
        end
        for i=1:m
            [~,Indx_G]=min(opt_G(:,i));
            G(:,i)=I1(:,Indx_G);
        end
    end

    % updating Z:
    % L_Z= ω \sum n\sum m ||P^Tx_i-u_j||^2_2 * Z_ij
    % + \sum n\sum m Z_ij * ||f_i-g_j||^2_2
    % + β ||Z||^2_F
    V = zeros(n,m);
    for i=1:n
        for j=1:m
            V(i,j) = omega*norm(P'*X(:,i)-U(:,j))^2+norm(F(:,i)-G(:,j))^2;
        end
    end
    [V11 Indx] = sort(V,2);
    for i=1:n
        beta1(i)=0.5*kk*V11(i,kk+1)-0.5*sum(V11(i,1:kk));
    end
    for i=1:n
        eta(i)=1/kk+sum(V11(i,1:kk))/(2*beta1(i)*kk);
    end
    Z = zeros(n,m);
    for i=1:n
        for j=1:kk
            Z(i,Indx(i,j))=-0.5*V11(i,j)/beta1(i)+eta(i);
        end
    end

    % updating F:
    % L_F= \sum n\sum m Z_ij * ||f_i-g_j||^2_2 
    % s.t. F\in Ind
    opt_F=zeros(class,ntu);
    for i=1:class
        for j=1:ntu
            row_Z = find(Z(j+nl,:));
            for k=1:length(row_Z)
                opt_F(i,j)=opt_F(i,j)+Z(j+nl,row_Z(k))*norm(I1(:,i)-G(:,row_Z(k)))^2;
            end
        end
    end
    for i=1:ntu
        [~,Indx_F]=min(opt_F(:,i));
        F(:,i+nl)=I1(:,Indx_F);
    end

end
end

function B = Construct_B(Y)
B = Y;
B(Y==0) = -1;
end
function Y = Construct_Y(X_label,n,Class)
Y = zeros(Class,n);
for i = 1:Class
    for j = 1:n
        if i == X_label(j)
            Y(i,j) = 1;
        end
    end
end
end
