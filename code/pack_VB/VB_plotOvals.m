function VB_plotOvals(model, label)
n = 361;
ang = linspace(0,2*pi, n);
circ = [cos(ang); sin(ang)];
[D,K] = size(model.m);

colr = 'brgmcyk';
nC = length(colr);
c = max(label);

for k = 1:K,
	scale = 2 / sqrt(model.alpha(k));
	scale2 = scale * sqrt(model.alpha(k)/500);
	ovum = scale * model.U(:,:,k)'*circ;
	ovum2 = scale2 * model.U(:,:,k)'*circ;
	plot(ovum(1,:)+model.m(1,k), ovum(2,:)+model.m(2,k), 'k-', 'color', [.8 .8 .8]); 
% colr(mod(k-1,nC)+1));
%	plot(ovum2(1,:)+model.m(1,k), ovum2(2,:)+model.m(2,k), 'k-', 'linewid', 2);
%	plot(model.m(1,k), model.m(2,k), 'k+', 'linewid', 2);
end

for k = 1:K,
	scale = 2 / sqrt(model.alpha(k));
	scale2 = scale * sqrt(model.alpha(k)/500);
	ovum = scale * model.U(:,:,k)'*circ;
	ovum2 = scale2 * model.U(:,:,k)'*circ;
%	plot(ovum(1,:)+model.m(1,k), ovum(2,:)+model.m(2,k), 'k-', 'color', [.8 .8 .8]); 
% colr(mod(k-1,nC)+1));
	plot(ovum2(1,:)+model.m(1,k), ovum2(2,:)+model.m(2,k), 'k-', 'linewid', 2);
	plot(model.m(1,k), model.m(2,k), 'k+', 'linewid', 2);
end
