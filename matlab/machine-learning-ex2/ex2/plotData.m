function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos = find(y==1); 
neg = find(y==0);

plot(X(pos, 1), X(pos, 2), "+k", "markersize", 7  );

plot(X(neg, 1), X(neg, 2), "or", "markersize", 7 );

xlabel("exam1 score");
ylabel("exam2 score");



% =========================================================================



hold off;

end
