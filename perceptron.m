 %x = [-1,0.5,0.5;-1,-0.5,-0.5;-1,0.5,-0.5;-1,-0.5,0.5];
 %y = [0.5;-0.5;0.5;0.5]; 
 x = [-1,1,1;-1,-1,-1;-1,1,-1;-1,-1,1];
 y = [1;-1;1;1];


function g = sigmoid(z)
%	Compute sigmoid function
  % You need to return the following variables correctly
  g = zeros(size(z));

  % Instructions: z can be a matrix, vector or scalar
  % g = 1.0 ./ ( 1.0 + exp(-z)); % For Matlab
  g = 1.0 ./ ( 1.0 + e.^(-z)); % For Octave, it can use 'exp(1)' or 'e'
    
end

function y = feed(input_x , w)
h = input_x * w;
%y = sign(h);
y = tanh(h);
endfunction




function dw = derivative(x, y , o)
y_derivative  =  (1 - y.*y);
dw = x' * ((o - y) .* y_derivative) ;
printf("derivative \n");
endfunction

function error = epoch_error(o_set, y_set)
error= mean((o_set - y_set).^2);
endfunction

function w = train(x_set, o_set, w, rate, epsilon)
error = epsilon;
epoch = 0;
while (epsilon <= error)
    epoch = epoch + 1;
    y_set = feed(x_set, w);
    error = epoch_error(o_set, y_set);
    dw = derivative(x_set, y_set, o_set);
    w = w + rate * dw;
    printf("epochs %d, error %f\n ",epoch, error)
end

endfunction

function w = ANN(input_size, output_size, learning_rate, x_set, o_set, epsilon)
w = randn(input_size,output_size);
w = train(x_set, o_set, w, learning_rate, epsilon);
endfunction









