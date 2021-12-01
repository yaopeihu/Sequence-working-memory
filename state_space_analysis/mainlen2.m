cd(foldername);
load(filename);
%% calculating the first(or the lowest) principal angles.
PA12 = getPrincipalAngle(weights_pool(:,1:6),weights_pool(:,7:12));

%% calculating the variance accounted for(VAF) ratio.
vaf_between = nan(2);
for i = 1:2
    for j = 1:2
        if i~=j
            B1 = weights_pool(:,(1:6)+(i-1)*6);
            B2 = weights_pool(:,(1:6)+(j-1)*6);
            vaf_between(i,j) =  getVAF(B1,B2);
        end
    end
end
%%
B = weights_pool(:,1:6);
[coeff,score1,latent,tsquared,explained,mu1] = pca(B','Algorithm','svd','Centered','on');
coeff1 = coeff(:,1:2);score1 = score1(:,1:2);
figure;plot((1:5)-0.2,cumsum(explained),'o-','markersize',10,'linewidth',2);ylim([0 100]);hold on
B = weights_pool(:,7:12);
[coeff,score2,latent,tsquared,explained,mu2] = pca(B','Algorithm','svd','Centered','on');
coeff2 = coeff(:,1:2);score2 = score2(:,1:2);
plot((1:5),cumsum(explained),'o-','markersize',10,'linewidth',2);ylim([0 100]);
box off
xlim([0.5 5.5])
xticks(gca,1:5)


cmap2 =  [10 108 176;
    239 124 33;
    52 153 57;
    202 42 40;
    139 100 168;
    138 85 74]/255;
figure;hold on;plot(score1([1:6,1],1),score1([1:6,1],2));scatter(score1(:,1),score1(:,2),200,cmap2,'filled');axis([-8 8 -8 8]);set(gca,'TickDir','out');axis('square')
figure;hold on;plot(score2([1:6,1],1),score2([1:6,1],2));scatter(score2(:,1),score2(:,2),200,cmap2,'filled');axis([-8 8 -8 8]);set(gca,'TickDir','out');axis('square')
%%
anchor_item = 2;
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


% check direction
check_item = [anchor_item+1 anchor_item-1];
st1 = sign(score1(check_item,2));
st2 = sign(score2(check_item,2));

if st1(1)>st1(2)
    coeff1 = fliplr(coeff1);
    score1 = fliplr(score1);
end

if st2(1)>st2(2)
    coeff2 = fliplr(coeff2);
    score2 = fliplr(score2);
end


anchor_item = 2;
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



%% initialization, the values of theta could be random, while the sigma should larger than 0.
theta1 = 0;
sigma1 = 0.8;
c1 = [sigma1*cos(theta1) -sigma1*sin(theta1);sigma1*sin(theta1) sigma1*cos(theta1)];
score11 = c1*score1';

theta2 = 0;sigma2 = 0.8;
c2 = [sigma2 *cos(theta2) -sigma2 *sin(theta2);sigma2 *sin(theta2) sigma2 *cos(theta2)];
score21 = c2*score2';


%
for i = 1:6
    for j = 1:2
        template(j,i) = mean([score11(j,i) score21(j,i)]);
    end
end

%%
myfun0 = @(theta,sigma)[sigma(1)*cos(theta) -sigma(1)*sin(theta);sigma(1)*sin(theta) sigma(1)*cos(theta)];
myfun1 = @(theta,sigma,template)norm(myfun0(theta(1),sigma(1))*score1'-reshape(template,2,6),'fro')+norm(myfun0(theta(2),sigma(2))*score2'-reshape(template,2,6),'fro');
myfun2 = @(theta,sigma)norm(myfun0(theta(1),sigma(1))*score1','fro')+norm(myfun0(theta(2),sigma(2))*score2','fro');
para_int = [theta1,theta2,sigma1,sigma2,reshape(template,1,[])];
[para,loss] = fmincon(@(a)myfun1(a(1:2),a(3:4),a(5:16))/myfun2(a(1:2),a(3:4)),para_int,[],[],[],[]);
loss

%%
 new_temp = reshape(para(5:end),2,6);
 figure;plot(template(1,[1:6,1]),template(2,[1:6,1]));hold on;plot(new_temp(1,[1:6,1]),new_temp(2,[1:6,1]));
s11 = myfun0(para(1),para(3))*score1';
s21 = myfun0(para(2),para(4))*score2';


figure;
scatter(s11(1,:),s11(2,:),200,cmap2,'filled');hold on;plot(s11(1,[1:6,1]),s11(2,[1:6,1]))
scatter(s21(1,:),s21(2,:),200,cmap2,'filled');hold on;plot(s21(1,[1:6,1]),s21(2,[1:6,1]))

hold on;plot(new_temp(1,[1:6,1]),new_temp(2,[1:6,1]),'k');

%%
Sim_score(1)=1-(norm(inv(myfun0(para(1),para(3)))*new_temp - score1','fro')/norm(score1,'fro')).^2
Sim_score(2)=1-(norm(inv(myfun0(para(2),para(4)))*new_temp - score2','fro')/norm(score2,'fro')).^2


%%
myfun3 = @(theta)[cos(theta) -sin(theta);sin(theta) cos(theta)];
s11 = myfun3(para(1))*score1';
s21 = myfun3(para(2))*score2';

figure;hold on;plot(s11(1,[1:6,1]),s11(2,[1:6,1]));scatter(s11(1,:),s11(2,:),200,cmap2,'filled');axis([-8 10 -10 8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out');axis square
figure;hold on;plot(s21(1,[1:6,1]),s21(2,[1:6,1]));scatter(s21(1,:),s21(2,:),200,cmap2,'filled');axis([-8 10 -10 8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out');axis square
%%
 for i = 1:2
    tmp(:,:,i) = [cos(para(i)) -sin(para(i));sin(para(i)) cos(para(i))];
 end
W1 = coeff1*inv(tmp(:,:,1));
W2 = coeff2*inv(tmp(:,:,2));
Q=[W1,W2];
lambda = inv(diag(para([3 3 4 4])));
 f = reshape(para(5:end),2,6);
 %%
 mi = 1./para([3 4]);scaler = mi(1);
 mi = mi/scaler;
 f_norm = f*scaler;
 
%   figure;plot(mi,'o-');xlim([0.7 2.3]);ylim([0.4 1.2]);yticks([.4 .6 .8 1]);box off
%   figure;hold on;plot(f_norm(1,[1:6,1]),f_norm(2,[1:6,1]));scatter(f_norm(1,:),f_norm(2,:),200,cmap2,'filled');axis([-8 10 -10 8]);xticks([-6 0 6]);yticks([-6 0 6]);set(gca,'Tickdir','out');axis square


%%




 weights_pool_model1 = Q*lambda*[f,zeros(2,6);zeros(2,6),f];

 model_ev11 =norm(weights_pool_model1','fro')/norm(weights_pool_reduce,'fro')

 save('encoding model.mat');






%% NPR,NSA


%% 
load('encoding model.mat')

for i = 1:size(W1,1)
L1(i) = norm(W1(i,:));
L2(i) = norm(W2(i,:));

end
%Q=Alignment
Q1 = L1.^2;
Q2 = L2.^2;




%% NPR
NPR1 = 1/length(L1)*(sum(L1.^2)).^2/sum(L1.^4);
NPR2 = 1/length(L2)*(sum(L2.^2)).^2/sum(L2.^4);
NPR = [NPR1,NPR2];save('NPR.mat',NPR);


%% NPR bootstrap across neurons£¨other kind of measurement on NPR£©
% boot_num = 100;neuron_num = length(L1);NPR=[];boot_size = neuron_num;
% for boot = 1:boot_num
%     rng shuffle
%     L1tmp = L1(randi([1,neuron_num],[boot_size 1]));
%     L2tmp = L2(randi([1,neuron_num],[boot_size 1]));
%     
%     
%     NPR(boot,1) = 1/boot_size * (sum(L1tmp.^2)).^2 / sum(L1tmp.^4);
%     NPR(boot,2) = 1/boot_size * (sum(L2tmp.^2)).^2 / sum(L2tmp.^4);
%    
% end
% 
% 
% edge = linspace(0.1,0.6,50);h = [];leng = {'Rank 1','Rank 2'};
% 
% prob1 = histcounts(NPR(:,1),edge,'Normalization','probability');
% prob2 = histcounts(NPR(:,2),edge,'Normalization','probability');
% 
% figure;hold on
% bar(edge(2:end),prob1,0.9,'EdgeColor','none');
% bar(edge(2:end),prob2,0.9,'EdgeColor','none');
% 
% xlim([0.25 0.55])
% median(NPR)


%%
neuron_num=length(W1);
thresh = round((mean(NPR)*neuron_num));

%%

iv12 = unique([iv1(1:thresh(1)),iv2(1:thresh(2))]);

edge = -1.1:0.1:1.1;
figure;
subplot(311);histogram((Q1(iv12)-Q2(iv12))./(Q1(iv12)+Q2(iv12)),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);xlabel('NSA(Rank1,Rank2)');xlim([-1.1 1.1]);ylim([0 0.1]);box off;set(gca,'Tickdir','out');ylabel('Proportion')

NSA_12 = (Q1(iv12)-Q2(iv12))./(Q1(iv12)+Q2(iv12));


% corrcoef(NSA_12,Q1(iv12))
% corrcoef(NSA_12,Q2(iv12))



%% rank modulation.
load('encoding model.mat', 'W1','W2');Q = [W1,W2];
% W1 = Q(:,1:2);
% W2 = Q(:,3:4);

for i = 1:size(W1,1)
L1(i) = norm(W1(i,:));
L2(i) = norm(W2(i,:));

end
%
Q1 = L1.^2;
Q2 = L2.^2;
[~,iv1] = sort(Q1,'descend');
[~,iv2] = sort(Q2,'descend');
load('NPR.mat');neuron_num = length(L1);
thresh = round(mean(NPR)*neuron_num);



%% get angular difference
theta = zeros(size(W1,1),2);
theta_diff = theta;
gain = theta;
%
for i = 1:2

theta(:,i)=acos(Q(:,2*(i-1)+1)./sqrt(Q(:,2*(i-1)+1).^2+Q(:,2*(i-1)+2).^2));

gain(:,i)=vecnorm(Q(:,2*(i-1)+(1:2)),2,2);
end
theta = (theta/pi*180);

%

 [~,iv1] = sort(L1,'descend');
[~,iv2] = sort(L2,'descend');

iv12 = intersect(iv1(1:thresh(1)),iv2(1:thresh(2)));

%

X = theta;
theta_diff1 = abs(X(:,1)-X(:,2));


edge = 0:1:10;



figure;
subplot(311);histogram(theta_diff1(iv12),edge,'Normalization','probability','EdgeColor','w','LineWidth',0.5);box off;xticks(edge);xlim([-1 11])
%%
load('\M2_len2.mat', 'weights_pool')


tuning_diff =zeros(size(weights_pool,1),1);
for  k = 1:size(weights_pool,1)
    neuron_rsp = reshape(weights_pool(k,:),6,2);
    [~,vtmp]=max(neuron_rsp,[],1);
    vtmp = abs(diff(vtmp));
    if vtmp>3
        vtmp = 6-vtmp;
%     elseif vtmp<-3
%         vtmp = 6+vtmp;
    elseif vtmp==3
        vtmp =0;
    end
    tuning_diff(k)=vtmp;
end

%%
figure;boxplot(theta_diff1(iv12),tuning_diff(iv12));ylim([-10 180])
%
[~,~,stat]=anovan(theta_diff1(iv12),tuning_diff(iv12),'display','off');
[c,m] = multcompare(stat,'Display','off')
%%
[shift_Q12_boot1,shift_Q12_boot2,lock_Q12_boot1,lock_Q12_boot2]=MI_theta(theta_diff1,Q1,Q2,iv12,{'1','2'});







