function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
lambda = 500;
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X, 2);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    h = theta' * X';
    for j = 1 : n
      T = X(:,j);
      temp = (h * T - y' * T) * alpha * (1 / m);
      % theta(j, 1) = theta(j, 1) - temp;
      if j == 1
          theta(j, 1) = theta(j, 1) - temp;
      else 
          theta(j, 1) = theta(j, 1) * (1 - alpha * lambda / m) - temp;
    end









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
