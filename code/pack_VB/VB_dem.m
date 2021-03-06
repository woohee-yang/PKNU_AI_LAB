%- Mo Chen -
%	- logMvGamma(X, d)
%	- logsumexp(X, dim)
%	- mixGaussRnd(d,k,n)
%	- mxiGausVb(X, m, prior)
%	- mivGaussVbPred(model, X)
%	- plotGauss(X, label)
%	mixGaussVb_demo

run_mode = 1;	%- show every step, or > 1 (batch)

%close all; clear;
% --------------- data ------------------------------------------------
d = 2;
k = 5;	% data set : # Gaussians
K = k;
n = 2000/2;
datachoice = 2-2*0;		%-  1: VB_mogrnd,    2: gamrnd+randn+randn -
if (datachoice == 1)
    [X,z] = VB_mogrnd(d,k,n);
elseif (datachoice == 2)
    %-- nk = [#samples(k)], 분배, sk = 각각의 산포도/분산(->std로?) --
    nk = rand(1,k); nk = round(nk / sum(nk) * n); nk(end) = n - sum(nk(1:end-1)); sum(nk)
%    Z = [];	%- no use ? -
    sk = gamrnd(10, 0.2, 1,k);		%-- common  variance^1 --
    sk
    Mk = randn (d, k)*4; X = []; z = [];	%-- k means, data buffers: X, z --
    switch d
    case 2
	for k = 1:k, X = [X , bsxfun(@plus, randn(d, nk(k))*sk(k), Mk(:,k))];
		z = [z, ones(1,nk(k))*k];
	end
    case 3,
	for k = 1:k, X = [X , bsxfun(@plus, randn(d, nk(k))*sk(k), Mk(:,k))];
		z = [z, ones(1,nk(k))*k];
	end
    end
    shuffle = randperm(n);
    X = X(:, shuffle);
    z = z(shuffle);
% save VB_dem_ex2 Xvb zvb modelVB y1 L_iter
%load VB_demo_ex2
else
	load VB_dem_ex2; X = Xvb; z = zvb;
end
m = floor(n/2);	%--> 데이터의 절반만 쓴다 --
X1 = X(:,1:m);	% d x m ?
X2 = X(:,(m+1):end);	% d x m ? -- for prediction

fi = 2+5;
% --------------- plot data -------------------------------------------
     figure(fi);
	%set(gcf, 'position', [405 385 560 420]);
	set(gcf, 'position', [405 385+180 400 245]);
%  VB_plotClass(X,z);
%  VB_plotClass(X,ones(1,n)); pause(0.2);
     VB_plotClass(X1,ones(1,m),'.');
	%ax0 = [-14.4 18 -10.2 10.7];
	ax0 = [-15 17.5 -10.2 10.7];
	ax0 = [ -15.5318   17.6536  -10.2516   11.1235];
	% ax = axis;
	axis(ax0);
     pause(0.2);

% --------------- VB fitting -----------------------------------------
if (run_mode > 1)
    % ------------- no animation -------
    [y1, model, L] = VB_mog (X1,10);	%- #clusters starting with 10 which can be a model.
            figure(fi+1);			%-- cluster ovals --
            set(gcf, 'position', [405+400+15, 385+180, 400, 245]);
else
    % --------- animation view ----------
    fprintf('Variational Bayesian Gaussian mixture: animation \n');
    [d, n] = size(X);

    prior = struct('alpha', 1+0, 'kappa', 1+5, 'm', mean(X,2), 'v', d+1, 'M', eye(d));  % M = inv(W);
    prior.logW = -2*sum(log(diag(chol(prior.M))));
    tol = 1e-8;
    maxiter = 2000;
	viewters = [1 5 10 20 50:50:2000]+1;
    viewters = [1:50, 55:5:500];
    L = -inf(1,maxiter);
    label = zeros(1,n);
            figure(fi+1);	set(gcf, 'position', [405+400+15, 385+180, 400, 245]);
            figure(fi+2);	set(gcf, 'position', [405+(400+15)*2, 385+180, 400, 245]);
    model = [];
    %-> model = K;	%-- 최대 #G's -> it will turn into model (a structure) --
    for iter = 2:maxiter,
        [y1, model, L_iter] = VB_mog1 (model, X1,10, prior);    %- #clusters starting with K=10
%->  [y1, model, L_iter] = VB_mog1 (X1, model, prior);    %- #clusters starting with K=10
        L(iter) = L_iter;

        if any(viewters == iter),
            figure(fi+1);		%-- cluster ovals evolution --
            VB_plotClass(X1,y1,'.');
            hold on; VB_plotOvals(model, y1); hold off;
            ax = ax0; 
            axis(ax0);
            set(text(mean(ax(1:2)), ax(3)+0.95*(ax(4)-ax(3)), sprintf('%d', iter-1)), ...
	 'horizontalalign', 'center');;

            figure(fi+2);		%-- weights histogram evolution --
            wtz = mean(model.R);
            bar(1:length(wtz), wtz); xlim([0 11]); ylim([0 1]);
            drawnow; pause(0.1);

            if (datachoice == 0), input(sprintf('%d> ', iter-1)); end
        end
        %-- convergence check --
        if abs(L(iter)-L(iter-1)) < tol*abs(L(iter));
	fprintf('* termination at iteration %d\n', iter); break;
        end
    end
end

% --------------- VB final touch ------------------------------------
figure(fi+1);		%-- cluster ovals -- went above -
            VB_plotClass(X1,y1,'.');
            hold on; VB_plotOvals(model, y1); hold off;
            ax = ax0; 
            axis(ax0);
            set(text(mean(ax(1:2)), ax(3)+0.95*(ax(4)-ax(3)), sprintf('%d', iter-1)), ...
	 'horizontalalign', 'center');;

figure(fi+2);	%-- weights histogram --
            if (run_mode > 1), set(gcf, 'position', [405+830, 385+180, 400, 245]); end
            wtz = mean(model.R);
            bar(1:length(wtz), wtz); xlim([0 11]); ylim([0 1]);

figure(fi+3);	%-- marginal likelihood, membership matrix --
	set(gcf, 'position', [1034 385+180-420+8 560 420-100]);
	L = L(2:iter);
	subplot(1,2,1); plot(L);
	subplot(1,2,2); imagesc(model.R);

%{
% Predict testing data
[y2, R] = VB_mogpred(model,X2);
%figure(fi+4);
figure(13);
            VB_plotClass(X1,y1,'.');
            hold on; VB_plotOvals(model, y1);
VB_plotClass(X2,y2, 'x'); 
            hold off;
%}