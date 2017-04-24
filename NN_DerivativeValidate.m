g=csvread('C:\Users\lenovo\Desktop\pree\code\stock_market.csv')
p= g(: , 1:5)
p=p'
Tc=g(: , 6)
Tc = round(Tc)
Tc=Tc'
T = ind2vec(Tc) 
net1 = newpnn(p,T);
Y = sim(net1,p);
Yc = vec2ind(Y)
[C,order] = confusionmat(Tc,Yc)
houseTargets=Tc
houseInputs=p
net = newff(houseInputs,houseTargets,5);
net = train(net,houseInputs,houseTargets);
% and testing parameters.
TestIndex=110;
TestRange=linspace(-8,8,500); %should be scaled appropriate for data

net_ActivateDeriv=10;
OutputOffset = 5;
NN_GradientFunction= net1;



% functions used 
Prime=@(X,Y) [(-Y(3)+4*Y(2)-3*Y(1))./(-X(3)+4*X(2)-3*X(1)),...
    (Y(3:end)-Y(1:end-2))./(X(3:end)-X(1:end-2)),...
	(3*Y(end)-4*Y(end-1)+Y(end-2))./(3*X(end)-4*X(end-1)+X(end-2))];
% Numerical center difference approximation, implies evenly spaced samples,
% but not structurally required


%% Set up truth comparison, single dimension

Base     =zeros(size(houseInputs,1),length(TestRange));
Base_Diff=zeros(size(houseInputs,1),length(TestRange));
ActivateDeriv=zeros(size(houseInputs,1),length(TestRange));
for alley=3:6%1:size(houseInputs,1); %abbreviated to not clutter document
% define the test points along the dimension specified by alley
TestPoints=repmat(houseInputs(:,TestIndex),1,length(TestRange));
TestPoints(alley,:)=TestPoints(alley,:)+TestRange(:).';

temp = sim(net_ActivateDeriv,TestPoints)-repmat(OutputOffset,1,size(TestPoints,2));
ActivateDeriv(alley,:)=temp(alley,:);

Base(alley,:)= sim(net,TestPoints);
Base_Diff(alley,:)=Prime(TestRange,Base(alley,:));

end
%% Activation function derivative check, symbolic
 x_s=sym('x_s');
% y_s= 2/(1+exp(-2*x_s))-1; %Eqn of hyperbolic tangent, from apply_transfer
% dy_s=diff(y_s,x_s); % Put into apply_transfer of modified file
% ddy_s=diff(dy_s,x_s); % Put into derivative of modified file 
% fprintf('Derivative expression:\n');
% pretty(dy_s);
% fprintf('Derivative expression, simplified:\n');
% pretty(simple(dy_s));
% fprintf('Second derivative expression:\n');
% pretty(ddy_s);
% fprintf('Second derivative expression, simplified:\n');
% pretty(simple(ddy_s));
%% Activation function derivative numerical check
% n1 = linspace(-5,5,1000);
% a1 = tansig(n1); % replace function with desired
% figure(165)
% subplot(311);
% plot(n1,a1)
% title('Original activation function');
% da1=Prime(n1,a1);
% da2=dtansig_0(n1);
% subplot(323);
% plot(n1,da1,'b.',n1,da2,'ro')
% title('First Derivative')
% legend('Numeric','By Function');
% subplot(324)
% plot(n1,da1-da2,'k.')
% title('Difference of 1st derivative')
% dda1=Prime(n1,da1);
% dda2=dtansig_0('dn',n1);
% subplot(325);
% plot(n1,dda1,'b.',n1,dda2,'ro')
% title('Second Derivative')
% legend('Numeric','By Function');
% subplot(326)
% plot(n1,dda1-dda2,'k.')
% title('Difference of 2nd derivative')
% % Note that magnitude of the difference is primarily due to numerical
% % approximation error. Decreasing the step size of n1 will decrease the
% % magnitude of the error. 