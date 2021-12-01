% This script is our main state space analysis for identified rank subspace with a approximation procedure. 

cd(foldername);
%%

load(fulltext(foldername,filename),'weights_pool');% get demo data.
%% calculating the first(or the lowest) principal angles.
PA12 = getPrincipalAngle(weights_pool(:,1:6),weights_pool(:,7:12));
PA13 = getPrincipalAngle(weights_pool(:,1:6),weights_pool(:,13:18));
PA23 = getPrincipalAngle(weights_pool(:,7:12),weights_pool(:,13:18));
%% calculating the variance accounted for(VAF) ratio.
vaf_between = nan(3);
for i = 1:3
    for j = 1:3
        if i~=j
            B1 = weights_pool(:,(1:6)+(i-1)*6);
            B2 = weights_pool(:,(1:6)+(j-1)*6);
            vaf_between(i,j) =  getVAF(B1,B2);
        end
    end
end
%% calculating vaf and the first principal angles on with bootstrapping across trials
load(fulltext(foldername,filename),'weights_pool0');
bootstrap_vaf_and_principal_angles(weights_pool0)
%%

B = weights_pool(:,1:6);
[coeff,score1,latent,tsquared,explained,mu1] = pca(B','Algorithm','svd','Centered','on');
coeff1 = coeff(:,1:2);score1 = score1(:,1:2);
figure;plot((1:5)-0.2,cumsum(explained),'o-','markersize',10,'linewidth',2);ylim([0 100]);hold on
B = weights_pool(:,7:12);
[coeff,score2,latent,tsquared,explained,mu2] = pca(B','Algorithm','svd','Centered','on');
coeff2 = coeff(:,1:2);score2 = score2(:,1:2);
plot((1:5),cumsum(explained),'o-','markersize',10,'linewidth',2);ylim([0 100]);
B = weights_pool(:,13:18);
[coeff,score3,latent,tsquared,explained,mu3] = pca(B','Algorithm','svd','Centered','on');
coeff3 = coeff(:,1:2);score3 = score3(:,1:2);
plot((1:5)+0.2,cumsum(explained),'o-','markersize',10,'linewidth',2);ylim([0 100]);
% box off
% xlim([0.5 5.5])
% xticks(gca,1:5)
% cmap2 = [0.00,0.45,0.74;
%     0.85,0.33,0.10;
%     0.93,0.69,0.13;
%     0.49,0.18,0.56;
%     0.47,0.67,0.19;
%     0.30,0.75,0.93];
cmap2 =  [10 108 176;
    239 124 33;
    52 153 57;
    202 42 40;
    139 100 168;
    138 85 74]/255;
%

% figure;scatter(score1(:,1),score1(:,2),200,cmap2,'filled');hold on;plot(score1([1:6,1],1),score1([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score2(:,1),score2(:,2),200,cmap2,'filled');hold on;plot(score2([1:6,1],1),score2([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score3(:,1),score3(:,2),200,cmap2,'filled');hold on;plot(score3([1:6,1],1),score3([1:6,1],2));axis([-8 8 -8 8]);

%% Initialization of approximation.
% Rotate the coordinates of a specific item across all ranks to 'PC1-axis'.
anchor_item = 3;
if score1(anchor_item,1)>0
    ra = -atan(score1(anchor_item,2)/score1(anchor_item,1));%/pi*180;
elseif score1(anchor_item,1)<0 
    ra = -pi-atan(score1(anchor_item,2)/score1(anchor_item,1));%/pi*180;
end

myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
score1 = myfun3(ra)*score1';
coeff1 = coeff1*inv(myfun3(ra));
score1 = score1';

score_tmp = score2;
coeff_tmp = coeff2;
if score_tmp(anchor_item,1)>0
    ra = -atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
elseif score_tmp(anchor_item,1)<0 
    ra = -pi-atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
end

myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
score_tmp = myfun3(ra)*score_tmp';
coeff_tmp = coeff_tmp*inv(myfun3(ra));

score_tmp = score_tmp';

score2 =score_tmp;
coeff2 = coeff_tmp;
%
score_tmp = score3;
coeff_tmp = coeff3;
if score_tmp(anchor_item,1)>0
    ra = -atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
elseif score_tmp(anchor_item,1)<0 
    ra = -pi-atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
end

myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
score_tmp = myfun3(ra)*score_tmp';
coeff_tmp = coeff_tmp*inv(myfun3(ra));

score_tmp = score_tmp';
tmp2 = coeff_tmp*score_tmp';
score3 =score_tmp;
coeff3 = coeff_tmp;

% check direction
check_item = [anchor_item-1 anchor_item+1];
st1 = sign(score1(check_item,2));
st2 = sign(score2(check_item,2));
st3 = sign(score3(check_item,2));
if st1(1)>st1(2)
    coeff1 = fliplr(coeff1);
    score1 = fliplr(score1);
end

if st2(1)>st2(2)
    coeff2 = fliplr(coeff2);
    score2 = fliplr(score2);
end

if st3(1)>st3(2)
    coeff3 = fliplr(coeff3);
    score3 = fliplr(score3);   
end


%

%
anchor_item = 3;
if score1(anchor_item,1)>0
    ra = -atan(score1(anchor_item,2)/score1(anchor_item,1));%/pi*180;
elseif score1(anchor_item,1)<0 
    ra = -pi-atan(score1(anchor_item,2)/score1(anchor_item,1));%/pi*180;
end

myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
score1 = myfun3(ra)*score1';
coeff1 = coeff1*inv(myfun3(ra));
% figure;scatter(s11(1,:),s11(2,:),200,cmap2,'filled');hold on;plot(s11(1,[1:6,1]),s11(2,[1:6,1]));axis([-8 8 -8 8]);
score1 = score1';
% tmp2 = coeff1*score1';
% figure;scatter(score1(:,1),score1(:,2),200,cmap2,'filled');hold on;plot(score1([1:6,1],1),score1([1:6,1],2));axis([-8 8 -8 8]);

%
score_tmp = score2;
coeff_tmp = coeff2;
if score_tmp(anchor_item,1)>0
    ra = -atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
elseif score_tmp(anchor_item,1)<0 
    ra = -pi-atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
end

myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
score_tmp = myfun3(ra)*score_tmp';
coeff_tmp = coeff_tmp*inv(myfun3(ra));
% figure;scatter(s11(1,:),s11(2,:),200,cmap2,'filled');hold on;plot(s11(1,[1:6,1]),s11(2,[1:6,1]));axis([-8 8 -8 8]);
score_tmp = score_tmp';
% tmp2 = coeff_tmp*score_tmp';
% figure;scatter(score_tmp(:,1),score_tmp(:,2),200,cmap2,'filled');hold on;plot(score_tmp([1:6,1],1),score_tmp([1:6,1],2));axis([-8 8 -8 8]);
score2 =score_tmp;
coeff2 = coeff_tmp;
%
score_tmp = score3;
coeff_tmp = coeff3;
if score_tmp(anchor_item,1)>0
    ra = -atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
elseif score_tmp(anchor_item,1)<0 
    ra = -pi-atan(score_tmp(anchor_item,2)/score_tmp(anchor_item,1));%/pi*180;
end

myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
score_tmp = myfun3(ra)*score_tmp';
coeff_tmp = coeff_tmp*inv(myfun3(ra));
% figure;scatter(s11(1,:),s11(2,:),200,cmap2,'filled');hold on;plot(s11(1,[1:6,1]),s11(2,[1:6,1]));axis([-8 8 -8 8]);
score_tmp = score_tmp';
tmp2 = coeff_tmp*score_tmp';
% figure;scatter(score_tmp(:,1),score_tmp(:,2),200,cmap2,'filled');hold on;plot(score_tmp([1:6,1],1),score_tmp([1:6,1],2));axis([-8 8 -8 8]);
score3 =score_tmp;
coeff3 = coeff_tmp;
%%
% figure;scatter(score1(:,1),score1(:,2),200,cmap2,'filled');hold on;plot(score1([1:6,1],1),score1([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score2(:,1),score2(:,2),200,cmap2,'filled');hold on;plot(score2([1:6,1],1),score2([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score3(:,1),score3(:,2),200,cmap2,'filled');hold on;plot(score3([1:6,1],1),score3([1:6,1],2));axis([-8 8 -8 8]);
%%
theta1 = pi/2;
sigma1 = 0.8;
c1 = [sigma1*cos(theta1) -sigma1*sin(theta1);sigma1*sin(theta1) sigma1*cos(theta1)];
score11 = c1*score1';

theta2 = pi/2;sigma2 = 0.8;
c2 = [sigma2 *cos(theta2) -sigma2 *sin(theta2);sigma2 *sin(theta2) sigma2 *cos(theta2)];
score21 = c2*score2';

theta3 = pi/2;
sigma3 = 1;
c3 = [sigma3*cos(theta3) -sigma3*sin(theta3);sigma3*sin(theta3) sigma3*cos(theta3)];
score31 = c3*score3';
template=[];
for i = 1:6
    for j = 1:2
        template(j,i) = mean([score11(j,i) score21(j,i)]);
    end
end

%%
myfun0 = @(theta,sigma)[sigma(1)*cos(theta) -sigma(1)*sin(theta);sigma(1)*sin(theta) sigma(1)*cos(theta)];
myfun1 = @(theta,sigma,template)norm(myfun0(theta(1),sigma(1))*score1'-reshape(template,2,6),'fro')+norm(myfun0(theta(2),sigma(2))*score2'-reshape(template,2,6),'fro')...
    +norm(myfun0(theta(3),sigma(3))*score3'-reshape(template,2,6),'fro');
myfun2 = @(theta,sigma)norm(myfun0(theta(1),sigma(1))*score1','fro')+norm(myfun0(theta(2),sigma(2))*score2','fro')+norm(myfun0(theta(3),sigma(3))*score3','fro');
para_int = [theta1,theta2,theta3,sigma1,sigma2,sigma3,reshape(template,1,[])];
[para,loss] = fmincon(@(a)myfun1(a(1:3),a(4:6),a(7:18))/myfun2(a(1:3),a(4:6)),para_int,[],[],[],[]);
loss

%% similarity scores.
 new_temp = reshape(para(7:end),2,6);

Sim_score(1)=1-(norm(inv(myfun0(para(1),para(4)))*new_temp - score1','fro')/norm(score1,'fro')).^2
Sim_score(2)=1-(norm(inv(myfun0(para(2),para(5)))*new_temp - score2','fro')/norm(score2,'fro')).^2
Sim_score(3)=1-(norm(inv(myfun0(para(3),para(6)))*new_temp - score3','fro')/norm(score3,'fro')).^2

%%
myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
s11 = myfun3(para(1))*score1';
s21 = myfun3(para(2))*score2';
s31 = myfun3(para(3))*score3';
%% coordinates in each rPC
figure;scatter(s11(1,:),s11(2,:),200,cmap2,'filled');hold on;plot(s11(1,[1:6,1]),s11(2,[1:6,1]));axis([-8 8 -8 8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out')
figure;scatter(s21(1,:),s21(2,:),200,cmap2,'filled');hold on;plot(s21(1,[1:6,1]),s21(2,[1:6,1]));axis([-8 8 -8 8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out')
figure;scatter(s31(1,:),s31(2,:),200,cmap2,'filled');hold on;plot(s31(1,[1:6,1]),s31(2,[1:6,1]));axis([-8 8 -8 8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out')
%%
 for i = 1:3
    tmp(:,:,i) = [cos(para(i)) -sin(para(i));sin(para(i)) cos(para(i))];
 end
W1 = coeff1*inv(tmp(:,:,1));
W2 = coeff2*inv(tmp(:,:,2));
W3 = coeff3*inv(tmp(:,:,3));

 f = reshape(para(7:end),2,6);
 lambda = inv(diag(para([4 4 5 5 6 6])));
 %
 mi = 1./para([4 5 6]);scaler = mi(1);
 mi = mi/scaler;
 f_norm = f*scaler;
 %
%   figure;plot(mi,'o-');xlim([0.7 3.3]);ylim([0.4 1.2]);yticks([.4 .6 .8 1])
%   figure;hold on;plot(f_norm(1,[1:6,1]),f_norm(2,[1:6,1]));scatter(f_norm(1,:),f_norm(2,:),200,cmap2,'filled');axis([-8.8 8.8 -8.8 8.8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out')



%%
 weights_pool_model1 = [W1,W2,W3]*lambda*[f,zeros(2,12);zeros(2,6),f,zeros(2,6);zeros(2,12),f];

 save('encoding model.mat');

%% get modulation index from the resampling pool.

bootstrap_approximation(weights_pool0)
figure;hold on;plot(f_norm(1,[1:6,1]),f_norm(2,[1:6,1]));scatter(f_norm(1,:),f_norm(2,:),200,cmap2,'filled');axis([-8 8 -8 8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out')

%% NPR,NSA
for kkkk =1

%% 
% load('encoding model.mat')

for i = 1:size(W1,1)
L1(i) = norm(W1(i,:));
L2(i) = norm(W2(i,:));
L3(i) = norm(W3(i,:));
end
%Q=Alignment
Q1 = L1.^2;
Q2 = L2.^2;
Q3 = L3.^2;

[~,iv1] = sort(L1,'descend');
[~,iv2] = sort(L2,'descend');
[~,iv3] = sort(L3,'descend');


%% NPR
NPR1 = 1/length(L1)*(sum(L1.^2)).^2/sum(L1.^4);
NPR2 = 1/length(L2)*(sum(L2.^2)).^2/sum(L2.^4);
NPR3 = 1/length(L3)*(sum(L3.^2)).^2/sum(L3.^4);
NPR = [NPR1,NPR2,NPR3];save('NPR.mat','NPR')
figure;bar([1 2 3],[NPR1,NPR2,NPR3]);xticklabels({'Rank 1','Rank 2','Rank 3'});
%% NPR bootstrap across neurons(another method to measure NPR)
% boot_num = 100;neuron_num = length(L1);NPR=[];boot_size = neuron_num;
% for boot = 1:boot_num
%     rng shuffle
%     L1tmp = L1(randi([1,neuron_num],[boot_size 1]));
%     L2tmp = L2(randi([1,neuron_num],[boot_size 1]));
%     L3tmp = L3(randi([1,neuron_num],[boot_size 1]));
%     
%     NPR(boot,1) = 1/boot_size * (sum(L1tmp.^2)).^2 / sum(L1tmp.^4);
%     NPR(boot,2) = 1/boot_size * (sum(L2tmp.^2)).^2 / sum(L2tmp.^4);
%     NPR(boot,3) = 1/boot_size * (sum(L3tmp.^2)).^2 / sum(L3tmp.^4);
% end
%

% edge = linspace(0.1,0.6,50);h = [];leng = {'Rank 1','Rank 2','Rank 3'};

% prob1 = histcounts(NPR(:,1),edge,'Normalization','probability');
% prob2 = histcounts(NPR(:,2),edge,'Normalization','probability');
% prob3 = histcounts(NPR(:,3),edge,'Normalization','probability');
% figure;hold on
% bar(edge(2:end),prob1,0.9,'EdgeColor','none');
% bar(edge(2:end),prob2,0.9,'EdgeColor','none');
% bar(edge(2:end),prob3,0.9,'EdgeColor','none');
% xlim([0.25 0.55])
% median(NPR)


%% get NSA
neuron_num=length(W1);
thresh = round((mean(NPR)*neuron_num))


iv12 = unique([iv1(1:thresh),iv2(1:thresh)]);
iv13 = unique([iv1(1:thresh),iv3(1:thresh)]);
iv23 = unique([iv2(1:thresh),iv3(1:thresh)]);
edge = -1.1:0.1:1.1;
figure;
subplot(311);histogram((Q1(iv12)-Q2(iv12))./(Q1(iv12)+Q2(iv12)),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);xlabel('NSA(Rank1, Rank2)');xlim([-1.1 1.1]);ylim([0 0.1]);box off;set(gca,'Tickdir','out')
subplot(312);histogram((Q1(iv13)-Q3(iv13))./(Q1(iv13)+Q3(iv13)),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);xlabel('NSA(Rank1, Rank3)');xlim([-1.1 1.1]);ylim([0 0.11]);box off;set(gca,'Tickdir','out')
subplot(313);histogram((Q2(iv23)-Q3(iv23))./(Q2(iv23)+Q3(iv23)),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);xlabel('NSA(Rank2, Rank3)');xlim([-1.1 1.1]);ylim([0 0.1]);box off;set(gca,'Tickdir','out')

NSA_12 = (Q1(iv12)-Q2(iv12))./(Q1(iv12)+Q2(iv12));
NSA_13 = (Q1(iv13)-Q3(iv13))./(Q1(iv13)+Q3(iv13));
NSA_23 = (Q2(iv23)-Q3(iv23))./(Q2(iv23)+Q3(iv23));

%%
save('rankmodulation_len3.mat');
end
%% phi difference

load('encoding model', 'W1','W2','W3');

Q = [W1,W2,W3];

neuron_num = size(W1,1);
for i = 1:neuron_num
    L1(i) = norm(W1(i,:));
    L2(i) = norm(W2(i,:));
    L3(i) = norm(W3(i,:));
end

theta = zeros(size(Q,1),2);
theta_diff = theta;


%%
for i = 1:3

theta(:,i)=acos(Q(:,2*(i-1)+2)./sqrt(Q(:,2*(i-1)+1).^2+Q(:,2*(i-1)+2).^2));


end
theta = (theta/pi*180);

%%
 

[~,iv1] = sort(L1,'descend');
[~,iv2] = sort(L2,'descend');
[~,iv3] = sort(L3,'descend');
load('NPR.mat');
% load('E:\M2GLM\len3_v6_11fovs\rankmodulationM2_len3.mat', 'thresh');
thresh = round(NPR*neuron_num);
iv12 = intersect(iv1(1:thresh(1)),iv2(1:thresh(2)));
iv13 = intersect(iv1(1:thresh(1)),iv3(1:thresh(3)));
iv23 = intersect(iv2(1:thresh(2)),iv3(1:thresh(3)));


%% distribution of theta_diff
X = theta;
theta_diff1 = abs(X(:,1)-X(:,2));
theta_diff2 = abs(X(:,1)-X(:,3));
theta_diff3 = abs(X(:,2)-X(:,3));

edge = 0:30:180;


%
figure;
subplot(311);histogram(theta_diff1(iv12),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);box off;xticks(edge);xlim([-30 210])
subplot(313);histogram(theta_diff2(iv13),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);box off;xticks(edge);xlim([-30 210])
subplot(312);histogram(theta_diff3(iv23),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);box off;xticks(edge);xlim([-30 210])
%%
% edge = 0:1:10;
% 
% 
% %
% figure;
% subplot(311);histogram(theta_diff1(iv12),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);box off;xticks(edge);xlim([-1 11])
% subplot(312);histogram(theta_diff2(iv13),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);box off;xticks(edge);xlim([-1 11])
% subplot(313);histogram(theta_diff3(iv23),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);box off;xticks(edge);xlim([-1 11])

%%
load('encoding model.mat', 'weights_pool')


%%
tuning_diff =zeros(size(weights_pool,1),3);
for  k = 1:size(weights_pool,1)
    neuron_rsp = reshape(weights_pool(k,:),6,3);
    [~,vtmp1]=max(neuron_rsp,[],1);
   difftmp=[];
    for i = 1:2
        for j = i+1:3
           
    vtmp = abs(diff(vtmp1([i,j])));
    if vtmp>3
        vtmp = 6-vtmp;
%     elseif vtmp<-3
%         vtmp = 6+vtmp;
    elseif vtmp==3
        vtmp =0;
    end
    difftmp = [difftmp vtmp];
    
        end
    end
    tuning_diff(k,:)=difftmp;
end

%% phi_diff vs tuning_diff

figure;boxplot([theta_diff1(iv12);theta_diff2(iv13);theta_diff3(iv23)],[tuning_diff(iv12,1);tuning_diff(iv13,2);tuning_diff(iv23,3)],'OutlierSize',1);ylim([-10 180]);

[~,~,stat]=anovan([theta_diff1(iv12);theta_diff2(iv13);theta_diff3(iv23)],[tuning_diff(iv12,1);tuning_diff(iv13,2);tuning_diff(iv23,3)],'display','off');

[c,m] = multcompare(stat,'display','off')

%% phase shift and lock's contribution, respectively
load('rankmodulationM1_len3.mat', 'Q1', 'Q2', 'Q3')



[shift_Q12_boot1,shift_Q12_boot2,lock_Q12_boot1,lock_Q12_boot2]=MI_theta(theta_diff1,Q1,Q2,iv12,{'1','2'});
[shift_Q13_boot1,shift_Q13_boot2,lock_Q13_boot1,lock_Q13_boot2]=MI_theta(theta_diff2,Q1,Q3,iv13,{'1','3'});
[shift_Q23_boot1,shift_Q23_boot2,lock_Q23_boot1,lock_Q23_boot2]=MI_theta(theta_diff3,Q2,Q3,iv23,{'2','3'});

save('rankmodulation_len3_phaseIm.mat')



