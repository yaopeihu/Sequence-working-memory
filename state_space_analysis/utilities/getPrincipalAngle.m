function PA = getPrincipalAngle(S1,S2)
% calculating the first principal angles 

[coeff,score1,~,~,~,~] = pca(S1','Algorithm','svd');
coeff1 = coeff(:,1:2);score1 = score1(:,1:2);

[coeff,score2,~,~,~,~] = pca(S2','Algorithm','svd');
coeff2 = coeff(:,1:2);score2 = score2(:,1:2);

[~,C12,~] = svd(coeff1'*coeff2);

%
PA = acos(C12(1))./pi*180;

end
