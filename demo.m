clc;clear;
addpath CMU-PIE/
%% CMU-PIE
src_str={'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgt_str={'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};
a = 1;
src = src_str{a};
tgt = tgt_str{a};
fprintf(' %s vs %s \n', src, tgt);
load(['CMU-PIE/' src '.mat']);
Xs = fea'; clear fea;
Xs_label = gnd;clear gnd;
load(['CMU-PIE/' tgt '.mat']);
Xt = fea'; clear fea;
Xt_label = gnd;clear gnd;
Xs = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]);
Xt = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);

load([src '_vs_' tgt '_' num2str(326) '.mat']);
alpha=1;
gamma=0.5;
omega=0.5;

for num = 1 : 20
    Xt_tl=Xt(:,R(num,:));
    Xt_tlabel=Xt_label(R(num,:),:);
    Indx=1:length(Xt_label);
    Indx(R(num,:))=[];
    Xt_ul=Xt(:,Indx);
    Xt_ulabel=Xt_label(Indx,:);
    Xst=[Xs Xt_tl];
    Yst=[Xs_label;Xt_tlabel];
    Class=length(unique(Xt_label));
    [P,U,F,G,Z,M] = ESSDA(Xs,Xs_label,Xt_tl,Xt_tlabel,Xt_ul,Xt_ulabel,alpha,gamma,omega);
    X_train = P'*Xst;
    Y_test  = P'*Xt_ul;
    mdl = fitcknn(X_train',Yst);
    cls = predict(mdl,Y_test');
    acc = sum(cls==Xt_ulabel)/length(Xt_ulabel);
    Acc(num)=acc*100;
end

