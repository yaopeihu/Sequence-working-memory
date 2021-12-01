%this script is a demonstration to show how approximation was done with each
%resampled GLM coefficients.
%After each approximation the modulation index, similarity score, and
%angles between neuron-i and rank-r subspaces were calculated and pooled
%together.

function bootstrap_approximation(weights_pool0)
%%

% fp = '*\data\neurondata';
% cd(fp)
% 
% load([fp,'\M2_len3.mat'],'weights_pool0');% select demo data.

[neuron_num,var_num,kfold,bootnum] = size(weights_pool0);
theta_pool = zeros(neuron_num,3,kfold,bootnum);
loss_pool = zeros(kfold,bootnum);
%%
modulation_index_pool=zeros(kfold,bootnum,3);
Sim_score_pool = zeros(2,100,3);
f_pool = zeros(2,bootnum,2,6);
s11_pool = f_pool;
s21_pool = f_pool;
s31_pool = f_pool;
%%
for booti = 1:bootnum
    for foldi = 1:2
        %%
weights_pool = weights_pool0(:,:,foldi,booti);

B = weights_pool(:,1:6);
[coeff,score1,latent,tsquared,explained,mu1] = pca(B','Algorithm','svd','Centered','on');
coeff1 = coeff(:,1:2);score1 = score1(:,1:2);
% figure;plot((1:5)-0.2,cumsum(explained),'o-','markersize',10,'linewidth',2);ylim([0 100]);hold on
B = weights_pool(:,7:12);
[coeff,score2,latent,tsquared,explained,mu2] = pca(B','Algorithm','svd','Centered','on');
coeff2 = coeff(:,1:2);score2 = score2(:,1:2);
% plot((1:5),cumsum(explained),'o-','markersize',10,'linewidth',2);ylim([0 100]);
B = weights_pool(:,13:18);
[coeff,score3,latent,tsquared,explained,mu3] = pca(B','Algorithm','svd','Centered','on');
coeff3 = coeff(:,1:2);score3 = score3(:,1:2);

cmap2 =  [10 108 176;
    239 124 33;
    52 153 57;
    202 42 40;
    139 100 168;
    138 85 74]/255;
%
% 
% figure;scatter(score1(:,1),score1(:,2),200,cmap2,'filled');hold on;plot(score1([1:6,1],1),score1([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score2(:,1),score2(:,2),200,cmap2,'filled');hold on;plot(score2([1:6,1],1),score2([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score3(:,1),score3(:,2),200,cmap2,'filled');hold on;plot(score3([1:6,1],1),score3([1:6,1],2));axis([-8 8 -8 8]);

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
% 
% figure;scatter(score1(:,1),score1(:,2),200,cmap2,'filled');hold on;plot(score1([1:6,1],1),score1([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score2(:,1),score2(:,2),200,cmap2,'filled');hold on;plot(score2([1:6,1],1),score2([1:6,1],2));axis([-8 8 -8 8]);
% figure;scatter(score3(:,1),score3(:,2),200,cmap2,'filled');hold on;plot(score3([1:6,1],1),score3([1:6,1],2));axis([-8 8 -8 8]);
%% check direction
check_item = [anchor_item-1 anchor_item+1];
st1 = (score1(check_item,2));
st2 = (score2(check_item,2));
st3 = (score3(check_item,2));
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


%%

%
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
sigma1 = 2;
c1 = [sigma1*cos(theta1) -sigma1*sin(theta1);sigma1*sin(theta1) sigma1*cos(theta1)];
score11 = c1*score1';

theta2 = pi/2;sigma2 = 1;
c2 = [sigma2 *cos(theta2) -sigma2 *sin(theta2);sigma2 *sin(theta2) sigma2 *cos(theta2)];
score21 = c2*score2';



theta3 = pi/2;
sigma3 = 1;
c3 = [sigma3*cos(theta3) -sigma3*sin(theta3);sigma3*sin(theta3) sigma3*cos(theta3)];
score31 = c3*score3';

%
for i = 1:6
    for j = 1:2
        template(j,i) = mean([score11(j,i) score21(j,i)]);
    end
end

%
myfun0 = @(theta,sigma)[sigma(1)*cos(theta) -sigma(1)*sin(theta);sigma(1)*sin(theta) sigma(1)*cos(theta)];
myfun1 = @(theta,sigma,template)norm(myfun0(theta(1),sigma(1))*score1'-reshape(template,2,6),'fro')+norm(myfun0(theta(2),sigma(2))*score2'-reshape(template,2,6),'fro')...
    +norm(myfun0(theta(3),sigma(3))*score3'-reshape(template,2,6),'fro');
myfun2 = @(theta,sigma)norm(myfun0(theta(1),sigma(1))*score1','fro')+norm(myfun0(theta(2),sigma(2))*score2','fro')+norm(myfun0(theta(3),sigma(3))*score3','fro');
para_int = [theta1,theta2,theta3,sigma1,sigma2,sigma3,reshape(template,1,[])];
[para,loss] = fmincon(@(a)myfun1(a(1:3),a(4:6),a(7:18))/myfun2(a(1:3),a(4:6)),para_int,[],[],[],[]);
loss_pool(foldi,booti)=loss;

%
 new_temp = reshape(para(7:end),2,6);

%% get similarity scores
Sim_score(1)=1-(norm(inv(myfun0(para(1),para(4)))*new_temp - score1','fro')/norm(score1,'fro')).^2;
Sim_score(2)=1-(norm(inv(myfun0(para(2),para(5)))*new_temp - score2','fro')/norm(score2,'fro')).^2;
Sim_score(3)=1-(norm(inv(myfun0(para(3),para(6)))*new_temp - score3','fro')/norm(score3,'fro')).^2;
Sim_score_pool(foldi,booti,:)=Sim_score;
%%




 for i = 1:3
    tmp(:,:,i) = [cos(para(i)) -sin(para(i));sin(para(i)) cos(para(i))];
 end
W1 = coeff1*inv(tmp(:,:,1));
W2 = coeff2*inv(tmp(:,:,2));
W3 = coeff3*inv(tmp(:,:,3));

%% get rank modulation index


 f = reshape(para(7:end),2,6);

  mi = 1./(para([4 5 6])/max(para([4 5 6])));
  f_norm = f/max(para([4 5 6]));
modulation_index_pool(foldi,booti,:)=1./(para([4 5 6]));%;mi;
f_pool(foldi,booti,:,:)=f;%_norm;



%% get phi values which measured the alignments between neuron and rank subspace.

Q = [W1,W2,W3];
theta = zeros(size(Q,1),3);

gain = theta;
%
for i = 1:3

theta(:,i)=acos(Q(:,2*(i-1)+1)./sqrt(Q(:,2*(i-1)+1).^2+Q(:,2*(i-1)+2).^2));

gain(:,i)=vecnorm(Q(:,2*(i-1)+(1:2)),2,2);
end
 theta = (theta/pi*180);

theta_pool(:,:,foldi,booti)=theta;
    end
    
    
end
%%
MI= squeeze(mean(modulation_index_pool,1));
% MI= squeeze(reshape(modulation_index_pool,[],3));
scaler = mean(MI);
MI_norm = MI./scaler(1);
% MI=
figure;errorbar([1 2 3],mean(MI_norm),std(MI_norm));xlim([0.7 3.3]);ylim([0.5 1.1]);yticks([ .6 .8 1]);xticks([1 2 3]);box off;set(gca,'Tickdir','out')


%%
% save('encoding model_boots.mat');
