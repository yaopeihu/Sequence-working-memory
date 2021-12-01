function vaf_ratio = getVAF(S1,S2)
%get the variance account for ratio
%projecting S1 to S2, then raito between the variance of projected S1 in S2
%and the variance of original S1. VAF ratio depends on the order of S1 and
%S2.

[coeff1,score1,~,~,~,~] = pca(S1','Algorithm','svd','Centered','on');
coeff1 = coeff1(:,1:2);score1 = score1(:,1:2);
[coeff2,score2,~,~,~,~] = pca(S2','Algorithm','svd','Centered','on');
coeff2 = coeff2(:,1:2);score2 = score2(:,1:2);

vaf_ratio = (norm(coeff2*coeff2'*coeff1*score1','fro')/norm(coeff1*score1','fro')).^2;

end

