 x = [  1,1;
        -1,-1;
        1,-1;
        -1,1];
 y = [   -1;
        -1;
         1;
         1];

% w_vector is a vector of weight matrices
% b_vector is a vector of biases vectors

function y = feed(input_x , w_vector, b_vector)
    h = input_x;
    for i = 1 : size(w_vector)(2)
        h = h * w_vector{i} + b_vector{i};
        y = tanh(h);
    end

endfunction

RANDOM = 1;
BIAS = 2;
ZERO = 0;

function a_vector = internal_feed(input_x , w_vector, b_vector, layers_vector)
    x_size = size(input_x)(1);
    w_size = size(w_vector)(2);

    a_vector =  cell(1, w_size + 1);
    for i = 1 : (w_size + 1)
        a_vector{i} = zeros([x_size, w_size]);
    end

    a_vector{1} = input_x;
    for i = 1 : w_size
        h = (a_vector{i} * w_vector{i}) + [b_vector{i};b_vector{i};b_vector{i};b_vector{i}];
        y = tanh(h);
        a_vector{i+1} = y;
    end
endfunction

function b_matrix = generate_bias_matrix(b_vector, amount_of_patterns)
    b_matrix = b_vector;
    if(amount_of_patterns > 1)
        for  i = 2:amount_of_patterns
        b_matrix = [b_matrix; b_vector];
        end
    endif
endfunction

function error = epoch_error(o_set, y_set)
    error= mean((o_set - y_set).^2);
endfunction

function delta = delta_weight(prev_delta, W, A)
    % derivative tanh
    f_prime = 1 - A .* A;

    delta = f_prime .* (prev_delta * W');
endfunction

function result = createArrays(arraySize, type)
RANDOM = 1;
BIAS = 2;
ZERO = 0;
% arraySize es un vector con las dimensiones de cada capa ej: [4,3,2, 3, 5]
% nArrays es el numero de capas a crear
    n_arrays = size(arraySize)(2) - 1;
    result = cell(1, n_arrays );
    for i = 1 : (n_arrays)
        switch (type)
            case (RANDOM)
                result{i} = randn([arraySize(i), arraySize(i + 1)]);
            case (BIAS)
                result{i} = (zeros([1, arraySize(i+1)]) - 1);
            otherwise
                
                result{i} = (zeros([arraySize(i), arraySize(i + 1)]));
        endswitch
    end
end

function [new_w_vector, new_b_vector] = back_propagate(w_vector, layers_vector, b_vector, y_set, y_hat, eta, a_vector, input)
    RANDOM = 1;
    BIAS = 2;
    ZERO = 0;
    delta = y_set - y_hat;
    deltas = createArrays(layers_vector, ZERO);
    layers_size = size(w_vector)(2);
    deltas{layers_size} = delta;

    for i = layers_size - 1 : -1 : 1
        deltas{i} = delta_weight(deltas{i + 1}, w_vector{i+1}, a_vector{i + 1});
    end 
    new_w_vector = w_vector;
    new_b_vector = b_vector;
    new_w_vector{1} = w_vector{1} - eta * (input') * deltas{1};
    new_b_vector{1} = b_vector{1} - eta * sum(deltas{1});
    for i = 1 : layers_size
        new_w_vector{i} = w_vector{i} + eta * (a_vector{i}') * deltas{i}; 
        new_b_vector{i} = b_vector{i} + eta * sum(deltas{i});
    end
    
endfunction

function [w_vector, new_b_vector] = fit(x_set, y_set, epsilon, eta, w_vector, b_vector, layers_vector)
    % tenemos que calcular primero la de la ultima capa
    a_vector = internal_feed(x_set, w_vector, b_vector, layers_vector);
    error = epoch_error(y_set, a_vector{end});
    epoch = 0;
    new_b_vector = b_vector;
    new_w_vector = w_vector;
    while( error > epsilon)
        [new_w_vector, new_b_vector] = back_propagate(new_w_vector, layers_vector, new_b_vector, y_set, a_vector{end}, eta, a_vector, x_set);
        a_vector = internal_feed(x_set, new_w_vector, new_b_vector, layers_vector);
        error = epoch_error(y_set, a_vector{end});
        epoch += 1;
        printf("%d epochs \t %f error (MSE). \n", epoch, error);
    end
    

endfunction


function [w_vector, b_vector] = create(layers_vector, eta, epsilon, x_set, y_set)
    RANDOM = 1;
    BIAS = 2;
    ZERO = 0;
    w_vector = createArrays(layers_vector, RANDOM);
    b_vector = createArrays(layers_vector, BIAS);

    [w_vector, b_vector] = fit(x_set, y_set, epsilon, eta, w_vector, b_vector, layers_vector);
endfunction