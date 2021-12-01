%% get VAF and Principal_angle at bootstrap level. the weights_pool0 here are the results of 100 times 2-fold cross validated GLM.
function bootstrap_vaf_and_principal_angles(weights_pool0)
principal_angles_between= zeros(size(weights_pool0,4),3);
principal_angles_within= zeros(size(weights_pool0,4),3);
principal_angles_between1 = principal_angles_between;
principal_angles_within1 = principal_angles_within;
for j = 1:size(weights_pool0,4)
weights_pool = weights_pool0(:,:,1,j);

B = weights_pool(:,1:6);
[coeff,score1,latent,tsquared,explained,mu] = pca(B','Algorithm','svd');
coeff1 = coeff(:,1:2);score1 = score1(:,1:2);



B = weights_pool(:,7:12);
[coeff,score2,latent,tsquared,explained,mu] = pca(B','Algorithm','svd');
coeff2 = coeff(:,1:2);score2 = score2(:,1:2);



B = weights_pool(:,13:18);
[coeff,score3,latent,tsquared,explained,mu] = pca(B','Algorithm','svd');
coeff3 = coeff(:,1:2);score3 = score3(:,1:2);


%
[P1,C12,P2] = svd(coeff1'*coeff2);C12_pool(:,:,j)=C12;
[P1,C13,P3] = svd(coeff1'*coeff3);C13_pool(:,:,j)=C13;
[P2,C23,P3] = svd(coeff2'*coeff3);C23_pool(:,:,j)=C23;
%
principal_angles_between1(j,:) = [acos(C12(1)) acos(C23(1)) acos(C13(1))]./pi*180;
principal_angles_between(j,:) = [acos(C12(1)) acos(C23(1)) acos(C13(1))]./pi*90 + [acos(C12(2)) acos(C23(2)) acos(C13(2))]./pi*90;
%%
for i = 1:3
    weights_pool1 = weights_pool0(:,:,1,j);

B = weights_pool1(:,(1:6)+(i-1)*6);
[coeff,score1,latent,tsquared,explained,mu] = pca(B','Algorithm','svd');
coeff11 = coeff(:,1:2);score11 = score1(:,1:2);

% BB = bsxfun(@minus,B,mean(B));
% [U,S,V]=svd(BB);
% coeff11 = U(:,1:2);

    weights_pool2 = weights_pool0(:,:,2,j);

B = weights_pool2(:,(1:6)+(i-1)*6);
[coeff,score1,latent,tsquared,explained,mu] = pca(B','Algorithm','svd');
coeff12 = coeff(:,1:2);score12 = score1(:,1:2);
% BB = bsxfun(@minus,B,mean(B));
% [U,S,V]=svd(BB);
% coeff12 = U(:,1:2);

[P11,C1,P12] = svd(coeff11'*coeff12);C1_pool(:,:,j)=C1;
principal_angles_within(j,i)=acos(C1(1))./pi*90+acos(C1(2))./pi*90;
principal_angles_within1(j,i)=acos(C1(1))./pi*180;
end



end
%%
vaf_betweeen_pool = [];
vaf_within_pool=[];
vaf_within_pool21 = [];


for j = 1:size(weights_pool0,4)
weights_pool = weights_pool0(:,:,1,j);



%%
for i = 1:3
    for k = 1:3
        
B1 = weights_pool(:,(1:6)+(i-1)*6);
B2 = weights_pool(:,(1:6)+(k-1)*6);

    
[coeff1,score1,latent,tsquared,explained,mu1] = pca(B1','Algorithm','svd','Centered','on');
coeff1 = coeff1(:,1:2);score1 = score1(:,1:2);
[coeff2,score2,latent,tsquared,explained,mu1] = pca(B2','Algorithm','svd','Centered','on');
coeff2 = coeff2(:,1:2);score2 = score2(:,1:2);
% BB1 = bsxfun(@minus,B1,mean(B1));
% [U1,S1,V1]=svd(BB1);
% coeff2 = U(:,1:2);
% sor_betweeen_pool(i,k,j)=norm(score1(:,1:2)*(coeff1(:,1:2))'*coeff2(:,1:2),'fro')/norm(score1(:,1:2),'fro');
% sor_3rank(i,j) = 1-norm(score1(:,1:2)*(coeff1(:,1:2))'*coeff2(:,1:2)-score1(:,1:2),'fro')/norm(score1(:,1:2),'fro');
vaf_betweeen_pool(i,k,j) = (norm(coeff2*coeff2'*coeff1*score1','fro')/norm(coeff1*score1','fro')).^2;

% sor_betweeen_pool(i,k,j)=trace(coeff2*coeff2'*(coeff1*coeff1'))/trace(coeff1*coeff1');
    end
end
%

%%
for i = 1:3
    weights_pool1 = weights_pool0(:,:,1,j);

B = weights_pool1(:,(1:6)+(i-1)*6);
[coeff,score1,latent,tsquared,explained,mu] = pca(B','Algorithm','svd');
coeff11 = coeff(:,1:2);score11 = score1(:,1:2);

    weights_pool2 = weights_pool0(:,:,2,j);

B = weights_pool2(:,(1:6)+(i-1)*6);
[coeff,score1,latent,tsquared,explained,mu] = pca(B','Algorithm','svd');
coeff12 = coeff(:,1:2);score12 = score1(:,1:2);
% sor_within_pool(j,i)=norm(score11*coeff11'*coeff12,'fro')/norm(score11,'fro');
vaf_within_pool(j,i)=(norm(coeff12*coeff12'*coeff11*score11','fro')/norm(coeff11*score11','fro')).^2;
vaf_within_pool21(j,i)=norm(coeff11*coeff11'*coeff12*score12','fro')/norm(coeff12*score12','fro').^2;
% sor_within_pool(j,i)= trace(coeff12*coeff12'*(coeff11*coeff11'))/trace(coeff11*coeff11');
end



end


%%
fc = zeros(3,3,3);
fc(:,1,1) = [0 0 0];
fc(:,1,2) = [185 68 32]/255;
fc(:,2,1) = [233 134 100]/255;
fc(:,2,2) = [.5 .5 .5];
fc(:,1,3) = [74 103 48]/255;
fc(:,3,1) = [149 199 80]/255;
fc(:,3,3) = [204 204 204]/255;
fc(:,3,2) = [5 107 175]/255;
fc(:,2,3) = [80 179 222]/255;
edge = 0:0.005:1;
figure;hold on;
for i = 1:3
    for j = 1:3
        if i==j
            h(i,j)=histogram(vaf_within_pool(:,i),edge,'Normalization','probability','EdgeColor','none','FaceColor',fc(:,i,j));
        else
        h(i,j)=histogram(vaf_betweeen_pool(i,j,:),edge,'Normalization','probability','EdgeColor','none','FaceColor',fc(:,i,j));
        end
        len{i,j} = ['VAF',num2str(i),'-',num2str(j)];
    end
end
%
legend(h(:),len(:));



%%
figure;errorbar(mean(principal_angles_between1),std(principal_angles_between1),'color',[0.85,0.33,0.10]);
hold on;
% errorbar(mean(squeeze(mean(principal_angles_within,1))),std(squeeze(mean(principal_angles_within,1))));xlim([0.7 3.3]);ylim([0 90])
title({'  1-2                                  2-3                               1-3'},'Color','red','fontsize',12);
errorbar(mean(principal_angles_within1),std(principal_angles_within1),'color',[.5 .5 .5]);xlim([0.7 3.3]);ylim([0 90]);xticks([1 2 3]);xticklabels({'1-1','2-2','3-3'})
%

%%


