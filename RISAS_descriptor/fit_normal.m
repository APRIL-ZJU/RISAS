function nor = fit_normal(data, interestpoint, show_graph)
% Fit a plane to the set of coordinates
% For a passed list of points in (x,y,z) cartesian coordinates,
% find the plane that best fits the data, the unit vector normal to that plane
% with an initial point at the average of the x, y, and z values.

	if nargin == 1
		show_graph = false;
	end

	for i = 1:3
		X = data;
		X(:, i) = 1;
        
        % check wherther x_m can be solved
		X_m = X' * X;       % (the transpose of X) * X = |X|
		if det(X_m) == 0
			can_solve(i) = 0;  % X_m is reversible
			continue
		end
		can_solve(i) = 1;

		% Construct and normalize the normal vector
		coeff = pinv(X_m) * X' * data(:, i);
		c_neg = -coeff;
		c_neg(i) = 1;
		coeff(i) = 1;
		n(:, i) = c_neg / norm(coeff); 
        
        [v,d] = eig(X_m);
        mmm=1;
    end

	if sum(can_solve) == 0
		error('Planar fit to the data caused a singular matrix.')
		return
	end

	% Calculating residuals for each fit
	center = mean(data);
	off_center = [data(:, 1)-center(1) data(:, 2)-center(2) data(:, 3)-center(3)];
	for i = 1:3
		if can_solve(i) == 0
			residual_sum(i) = NaN;
			continue
		end

		residuals = off_center * n(:, i);
		residual_sum(i) = sum(residuals .* residuals);

	end

	% Find the lowest index
	best_fit = find(residual_sum == min(residual_sum));

	% Possible that equal mins so just use the first index found
	nor = n(:, best_fit(1));

	if ~show_graph
		return
	end

	range = max(max(data) - min(data)) / 2;
	mid_pt = (max(data) - min(data)) / 2 + min(data);
	xlim = [-1 1]*range + mid_pt(1);
	ylim = [-1 1]*range + mid_pt(2);
	zlim = [-1 1]*range + mid_pt(3);
    
    figure(4)
	L = plot3(data(:, 1), data(:, 2), data(:, 3), 'bo', 'Markerfacecolor', 'b'); % Plot the original data points
    hold on;
    L = plot3(interestpoint(1, 1), interestpoint(1, 2), interestpoint(1, 3), 'ro', 'Markerfacecolor', 'r'); % Plot the original data points
	hold on;
	set(get(L, 'Parent'), 'DataAspectRatio', [1 1 1], 'XLim', xlim, 'YLim', ylim, 'ZLim', zlim);

	norm_data = [interestpoint; interestpoint + (n' * range)];%%%20150624

	% Plot the original data points
	L = plot3(norm_data(:, 1), norm_data(:, 2), norm_data(:, 3), 'r-', 'LineWidth', 3);
	set(get(get(L, 'parent'), 'XLabel'), 'String', 'x', 'FontSize', 14, 'FontWeight', 'bold')
	set(get(get(L, 'parent'), 'YLabel'), 'String', 'y', 'FontSize', 14, 'FontWeight', 'bold')
	set(get(get(L, 'parent'), 'ZLabel'), 'String', 'z', 'FontSize', 14, 'FontWeight', 'bold')
	title(sprintf('Normal Vector: <%0.3f,  %0.3f,  %0.3f>', n), 'FontWeight', 'bold', 'FontSize', 14)
	grid on;
	axis square;
    % hold off;
end
